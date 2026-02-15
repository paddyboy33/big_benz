"""
MT5 Semi-Auto Trading Bot - XAUUSD M1 (Asia/Thai Session Only)
================================================================
Mode: Semi-Auto
- User sets direction via PENDING ORDER (BUY or SELL)
- Pending BUY = only take BUY signals (Higher Low)
- Pending SELL = only take SELL signals (Lower High)
- Pending order acts as a marker (not deleted after trade)

Timeframe: M1
Symbol: XAUUSDm only
Trading Hours: 08:00 - 18:00 Thailand time (no New York session)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, List

# ============================================================================
# CONFIGURATION
# ============================================================================

MT5_CONFIG = {
    'login': 413331734,
    'password': 'Benz1212312121*',
    'server': 'Exness-MT5Trial6',
    'path': r"C:\Users\User\AppData\Roaming\Bot3_STO_m1\terminal64.exe",
}

# Trading Settings
SYMBOL = 'XAUUSDm'
ACCOUNT_SIZE = 500
RISK_PERCENT = 10
FIX_LOT = 0.01  # Fixed lot size
LOOKBACK = 10
MAGIC_NUMBER = 20250210
TIMEFRAME = mt5.TIMEFRAME_M1
RR_RATIO = 1.5

# Trading Hours (Thailand time, local time on this machine)
TRADE_HOUR_START = 8   # 08:00
TRADE_HOUR_END = 18    # 18:00

CHECK_INTERVAL = 15  # Check every 15 seconds for M1

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('mt5_semi_auto_m1.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MT5 CONNECTION
# ============================================================================

class MT5Connection:
    def __init__(self, config: dict):
        self.config = config
        self.connected = False

    def connect(self) -> bool:
        if not mt5.initialize(path=self.config.get('path')):
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        authorized = mt5.login(
            login=self.config['login'],
            password=self.config['password'],
            server=self.config['server']
        )

        if not authorized:
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

        account_info = mt5.account_info()
        logger.info(f"Connected to MT5: {account_info.name}")
        logger.info(f"Account: {account_info.login} | Balance: ${account_info.balance:,.2f}")

        self.connected = True
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")

# ============================================================================
# TRADING HOURS
# ============================================================================

def is_trading_hours() -> bool:
    """Check if current time is within trading hours (Thai time)"""
    now = datetime.now()
    return TRADE_HOUR_START <= now.hour < TRADE_HOUR_END

def get_session_status() -> str:
    """Get current session status"""
    now = datetime.now()
    hour = now.hour
    if TRADE_HOUR_START <= hour < TRADE_HOUR_END:
        return f"ACTIVE (Thai {now.strftime('%H:%M')})"
    else:
        return f"CLOSED (Thai {now.strftime('%H:%M')}, trading {TRADE_HOUR_START:02d}:00-{TRADE_HOUR_END:02d}:00)"

# ============================================================================
# DIRECTION MANAGER
# ============================================================================

class DirectionManager:
    def get_allowed_direction(self, symbol: str) -> Optional[str]:
        orders = mt5.orders_get(symbol=symbol)

        if orders is None or len(orders) == 0:
            return None

        for order in orders:
            if order.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP]:
                return 'BUY'
            elif order.type in [mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP]:
                return 'SELL'

        return None

# ============================================================================
# SIGNAL DETECTION
# ============================================================================

class SignalDetector:
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.ovs_history = []
        self.ovb_history = []
        self.last_signal_type = None
        self.initialized = False
        self.last_processed_time = None

    def get_historical_data(self, bars: int = 500) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, bars)

        if rates is None or len(rates) == 0:
            logger.warning(f"Failed to get data for {SYMBOL}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        k_period = 9
        d_period = 3
        smooth_k = 3

        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()

        stoch_k_raw = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        df['sto_k'] = stoch_k_raw.rolling(window=smooth_k).mean()
        df['sto_d'] = df['sto_k'].rolling(window=d_period).mean()
        df['sto_k_prev'] = df['sto_k'].shift(1)

        df['oversold'] = (df['sto_k_prev'] < 20) & (df['sto_k'] >= 20)
        df['overbought'] = (df['sto_k_prev'] > 80) & (df['sto_k'] <= 80)

        df['rolling_low'] = df['low'].rolling(window=self.lookback).min()
        df['rolling_high'] = df['high'].rolling(window=self.lookback).max()

        return df

    def backfill_history(self, df: pd.DataFrame):
        logger.info(f"  Backfilling history for {SYMBOL}...")

        history_bars = df.iloc[:-1]
        signals_found = 0

        for idx in range(len(history_bars)):
            bar = history_bars.iloc[idx]
            bar_time = history_bars.index[idx]

            if bar['oversold']:
                signal = {
                    'datetime': bar_time,
                    'type': 'ovs',
                    'value': bar['rolling_low'],
                    'close': bar['close'],
                }

                if self.last_signal_type != 'ovs':
                    self.ovs_history.append(signal)
                    self.last_signal_type = 'ovs'
                    signals_found += 1
                else:
                    if self.ovs_history and signal['value'] < self.ovs_history[-1]['value']:
                        self.ovs_history[-1] = signal

            elif bar['overbought']:
                signal = {
                    'datetime': bar_time,
                    'type': 'ovb',
                    'value': bar['rolling_high'],
                    'close': bar['close'],
                }

                if self.last_signal_type != 'ovb':
                    self.ovb_history.append(signal)
                    self.last_signal_type = 'ovb'
                    signals_found += 1
                else:
                    if self.ovb_history and signal['value'] > self.ovb_history[-1]['value']:
                        self.ovb_history[-1] = signal

        logger.info(f"  {SYMBOL}: Found {signals_found} signal groups")
        logger.info(f"  {SYMBOL}: OVS: {len(self.ovs_history)}, OVB: {len(self.ovb_history)}")

        if self.ovs_history:
            last = self.ovs_history[-1]
            logger.info(f"  Last OVS: {last['datetime']}, value={last['value']:.2f}")
        if self.ovb_history:
            last = self.ovb_history[-1]
            logger.info(f"  Last OVB: {last['datetime']}, value={last['value']:.2f}")

    def check_signal(self, allowed_direction: str) -> Optional[Dict]:
        df = self.get_historical_data()
        if df is None or len(df) < 100:
            return None

        df = self.calculate_indicators(df)

        # Backfill on first run
        if not self.initialized:
            self.backfill_history(df)
            self.initialized = True
            self.last_processed_time = df.index[-2]
            return None

        latest = df.iloc[-2]
        current_time = df.index[-2]

        if self.last_processed_time and current_time <= self.last_processed_time:
            return None

        self.last_processed_time = current_time

        # OVS crossover
        if latest['oversold']:
            signal = {
                'datetime': current_time,
                'type': 'ovs',
                'value': latest['rolling_low'],
                'close': latest['close'],
            }

            if self.last_signal_type != 'ovs':
                self.ovs_history.append(signal)
                self.last_signal_type = 'ovs'

                if allowed_direction == 'BUY' and len(self.ovs_history) >= 2:
                    current = self.ovs_history[-1]
                    prev = self.ovs_history[-2]

                    if current['value'] > prev['value']:
                        logger.info(f"  {SYMBOL}: Higher Low! {prev['value']:.2f} -> {current['value']:.2f}")
                        return {
                            'type': 'BUY',
                            'symbol': SYMBOL,
                            'entry': current['close'],
                            'sl': prev['value'],
                            'signal_time': current_time
                        }
            else:
                if self.ovs_history and signal['value'] < self.ovs_history[-1]['value']:
                    self.ovs_history[-1] = signal

        # OVB crossover
        elif latest['overbought']:
            signal = {
                'datetime': current_time,
                'type': 'ovb',
                'value': latest['rolling_high'],
                'close': latest['close'],
            }

            if self.last_signal_type != 'ovb':
                self.ovb_history.append(signal)
                self.last_signal_type = 'ovb'

                if allowed_direction == 'SELL' and len(self.ovb_history) >= 2:
                    current = self.ovb_history[-1]
                    prev = self.ovb_history[-2]

                    if current['value'] < prev['value']:
                        logger.info(f"  {SYMBOL}: Lower High! {prev['value']:.2f} -> {current['value']:.2f}")
                        return {
                            'type': 'SELL',
                            'symbol': SYMBOL,
                            'entry': current['close'],
                            'sl': prev['value'],
                            'signal_time': current_time
                        }
            else:
                if self.ovb_history and signal['value'] > self.ovb_history[-1]['value']:
                    self.ovb_history[-1] = signal

        return None

# ============================================================================
# ORDER MANAGEMENT
# ============================================================================

class OrderManager:
    def __init__(self, magic_number: int, risk_percent: float):
        self.magic_number = magic_number
        self.risk_percent = risk_percent

    def get_symbol_info(self):
        info = mt5.symbol_info(SYMBOL)
        if info is None:
            logger.error(f"Symbol {SYMBOL} not found")
            return None
        if not info.visible:
            if not mt5.symbol_select(SYMBOL, True):
                logger.error(f"Failed to select {SYMBOL}")
                return None
        return info

    def calculate_lot_size(self, order_type: str, entry_price: float,
                          sl_price: float, account_balance: float) -> tuple:
        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            return 0.0, 0.0, False

        risk_amount = account_balance * (self.risk_percent / 100)

        if order_type == 'BUY':
            action = mt5.ORDER_TYPE_BUY
        else:
            action = mt5.ORDER_TYPE_SELL

        profit_1lot = mt5.order_calc_profit(action, SYMBOL, 1.0, entry_price, sl_price)

        if profit_1lot is None:
            logger.error(f"  order_calc_profit failed for {SYMBOL}")
            return symbol_info.volume_min, 0.0, False

        loss_per_lot = abs(profit_1lot)

        if loss_per_lot > 0:
            lots = risk_amount / loss_per_lot
        else:
            lots = symbol_info.volume_min

        lot_step = symbol_info.volume_step
        lots = round(lots / lot_step) * lot_step
        lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))

        expected_profit = mt5.order_calc_profit(action, SYMBOL, lots, entry_price, sl_price)
        expected_loss = abs(expected_profit) if expected_profit else 0

        is_valid = expected_loss <= risk_amount * 2

        logger.info(f"  Lot: {lots} | Risk: ${risk_amount:.2f} | Expected Loss: ${expected_loss:.2f}")

        return lots, expected_loss, is_valid

    def has_position(self) -> bool:
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions:
            for pos in positions:
                if pos.magic == self.magic_number:
                    return True
        return False

    def get_position(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions:
            for pos in positions:
                if pos.magic == self.magic_number:
                    return pos
        return None

    def open_position(self, order_type: str, entry: float, sl: float,
                      rr_ratio: float) -> bool:
        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            return False

        if order_type == 'BUY':
            risk_points = entry - sl
            tp = entry + (risk_points * rr_ratio)
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(SYMBOL).ask
        else:
            risk_points = sl - entry
            tp = entry - (risk_points * rr_ratio)
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(SYMBOL).bid

        lots = FIX_LOT
        logger.info(f"  Fixed Lot: {lots}")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": lots,
            "type": order_type_mt5,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": f"SemiM1_{order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")
            return False

        logger.info(f"ORDER OPENED: {order_type} {SYMBOL}")
        logger.info(f"  Lots: {lots:.2f} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        logger.info(f"  RR: 1:{rr_ratio}")

        return True

# ============================================================================
# MAIN BOT
# ============================================================================

class SemiAutoM1Bot:
    def __init__(self):
        self.mt5_conn = MT5Connection(MT5_CONFIG)
        self.direction_mgr = DirectionManager()
        self.signal_detector = SignalDetector(LOOKBACK)
        self.order_manager = OrderManager(MAGIC_NUMBER, RISK_PERCENT)
        self.running = False

    def start(self):
        logger.info("=" * 60)
        logger.info("SEMI-AUTO BOT - XAUUSD M1 (Thai Session)")
        logger.info("=" * 60)
        logger.info(f"Symbol: {SYMBOL}")
        logger.info(f"Timeframe: M1")
        logger.info(f"Trading Hours: {TRADE_HOUR_START:02d}:00 - {TRADE_HOUR_END:02d}:00 (Thai time)")
        logger.info(f"RR: 1:{RR_RATIO}")
        logger.info(f"Risk: {RISK_PERCENT}% of ${ACCOUNT_SIZE}")
        logger.info("Direction: Controlled by PENDING ORDERS")
        logger.info("=" * 60)

        if not self.mt5_conn.connect():
            logger.error("Failed to connect to MT5. Exiting.")
            return

        info = mt5.symbol_info(SYMBOL)
        if info is None:
            logger.error(f"{SYMBOL} not found. Exiting.")
            self.mt5_conn.disconnect()
            return

        logger.info(f"[OK] {SYMBOL}")

        # Show direction
        direction = self.direction_mgr.get_allowed_direction(SYMBOL)
        if direction:
            logger.info(f"Direction: {direction} signals allowed")
        else:
            logger.info(f"Direction: NO PENDING - waiting for pending order")

        logger.info(f"Session: {get_session_status()}")
        logger.info("\nBot is running. Press Ctrl+C to stop.\n")

        self.running = True

        try:
            self.main_loop()
        except KeyboardInterrupt:
            logger.info("\nStopping bot...")
        finally:
            self.stop()

    def main_loop(self):
        last_status_time = datetime.now() - timedelta(minutes=5)

        while self.running:
            try:
                now = datetime.now()

                # Status every 5 minutes
                if (now - last_status_time).seconds >= 300:
                    self.print_status()
                    last_status_time = now

                # Check trading hours
                if not is_trading_hours():
                    time.sleep(30)
                    continue

                # Check direction from pending order
                allowed_direction = self.direction_mgr.get_allowed_direction(SYMBOL)
                if allowed_direction is None:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Skip if already have position
                if self.order_manager.has_position():
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Check signal
                signal = self.signal_detector.check_signal(allowed_direction)

                if signal:
                    logger.info(f"\n{'=' * 50}")
                    logger.info(f"SIGNAL: {signal['type']} {SYMBOL}")
                    logger.info(f"Direction: {allowed_direction}")
                    logger.info(f"Entry: {signal['entry']:.2f} | SL: {signal['sl']:.2f}")
                    logger.info(f"Time: {now.strftime('%H:%M:%S')} (Thai)")
                    logger.info(f"{'=' * 50}")

                    success = self.order_manager.open_position(
                        order_type=signal['type'],
                        entry=signal['entry'],
                        sl=signal['sl'],
                        rr_ratio=RR_RATIO,
                    )

                    if success:
                        logger.info(f"Position opened!")
                    else:
                        logger.error(f"Failed to open position")

                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(10)

    def print_status(self):
        account = mt5.account_info()
        direction = self.direction_mgr.get_allowed_direction(SYMBOL)
        pos = self.order_manager.get_position()

        logger.info(f"\n--- STATUS ({datetime.now().strftime('%H:%M:%S')}) ---")
        logger.info(f"Balance: ${account.balance:,.2f} | Equity: ${account.equity:,.2f}")
        logger.info(f"Session: {get_session_status()}")
        logger.info(f"Direction: {direction if direction else 'NO PENDING'}")

        if pos:
            profit_str = f"+${pos.profit:.2f}" if pos.profit >= 0 else f"-${abs(pos.profit):.2f}"
            side = "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT"
            logger.info(f"Position: {side} {pos.volume} lots | PnL: {profit_str}")
        else:
            logger.info(f"Position: None - waiting for signal")

        logger.info("-" * 40)

    def stop(self):
        self.running = False
        self.mt5_conn.disconnect()
        logger.info("Bot stopped.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("""
    ================================================================
    SEMI-AUTO BOT - XAUUSD M1 (Thai Session Only)
    ================================================================
    How to use:
    1. Set PENDING ORDER on XAUUSDm (BUY LIMIT/STOP or SELL LIMIT/STOP)
    2. Pending BUY  = Bot takes BUY signals only (Higher Low)
       Pending SELL = Bot takes SELL signals only (Lower High)
    3. No pending   = Bot waits
    4. Trading hours: 08:00 - 18:00 Thai time only
    5. Pending order is NOT deleted - just a direction marker
    ================================================================
    """)

    bot = SemiAutoM1Bot()
    bot.start()
