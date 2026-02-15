"""
MT5 Semi-Auto Trading Bot - Stochastic Higher Low / Lower High Strategy
=========================================================================
Mode: Semi-Auto
- User sets direction via PENDING ORDER (BUY or SELL)
- Bot only trades in the direction of the pending order
- Pending BUY = only take BUY signals (Higher Low)
- Pending SELL = only take SELL signals (Lower High)
- Pending order acts as a marker (not deleted after trade)

Timeframe: M15
Risk: 0.5% per trade per symbol
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
    'login': 433143035,
    'password': 'Benz1212312121*',
    'server': 'Exness-MT5Trial7',
    'path': r"C:\Users\User\AppData\Roaming\MetaTrader 5 EXNESS\terminal64.exe",
}

# Trading Settings
ACCOUNT_SIZE = 500
RISK_PERCENT = 10
LOOKBACK = 10
MAGIC_NUMBER = 20250127  # Different magic for semi-auto bot
TIMEFRAME = mt5.TIMEFRAME_M15  # M15 timeframe

# RR per symbol (Exness naming)
SYMBOL_CONFIG = {
    'XAUUSDm': {'rr': 1.5},
    'US30m': {'rr': 1.5},
    'UKOILm': {'rr': 1.5},
    'USTECm': {'rr': 1.5},
}

CHECK_INTERVAL = 30  # Check every 30 seconds for M15

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('mt5_semi_auto.log', encoding='utf-8'),
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
# PENDING ORDER CHECKER (Direction Marker)
# ============================================================================

class DirectionManager:
    """
    Check pending orders to determine allowed trade direction per symbol.
    Pending BUY (LIMIT/STOP) = Allow BUY only
    Pending SELL (LIMIT/STOP) = Allow SELL only
    No pending = No trading for that symbol
    """

    def __init__(self, magic_number: int):
        self.magic_number = magic_number

    def get_allowed_direction(self, symbol: str) -> Optional[str]:
        """
        Check pending orders for symbol and return allowed direction.
        Returns: 'BUY', 'SELL', or None (no trading)
        """
        orders = mt5.orders_get(symbol=symbol)

        if orders is None or len(orders) == 0:
            return None

        # Check all pending orders for this symbol
        for order in orders:
            # BUY LIMIT or BUY STOP
            if order.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP]:
                return 'BUY'
            # SELL LIMIT or SELL STOP
            elif order.type in [mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP]:
                return 'SELL'

        return None

    def get_all_directions(self, symbols: list) -> Dict[str, str]:
        """Get allowed directions for all symbols"""
        directions = {}
        for symbol in symbols:
            direction = self.get_allowed_direction(symbol)
            if direction:
                directions[symbol] = direction
        return directions

# ============================================================================
# SIGNAL DETECTION
# ============================================================================

class SignalDetector:
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.ovs_history: Dict[str, List] = {}
        self.ovb_history: Dict[str, List] = {}
        self.last_signal_type: Dict[str, str] = {}
        self.initialized: Dict[str, bool] = {}
        self.last_processed_time: Dict[str, datetime] = {}

    def get_historical_data(self, symbol: str, bars: int = 500) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, bars)

        if rates is None or len(rates) == 0:
            logger.warning(f"Failed to get data for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Stochastic (9,3,3) - manual calculation
        k_period = 9
        d_period = 3
        smooth_k = 3

        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()

        stoch_k_raw = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        df['sto_k'] = stoch_k_raw.rolling(window=smooth_k).mean()
        df['sto_d'] = df['sto_k'].rolling(window=d_period).mean()
        df['sto_k_prev'] = df['sto_k'].shift(1)

        # Crossovers
        df['oversold'] = (df['sto_k_prev'] < 20) & (df['sto_k'] >= 20)
        df['overbought'] = (df['sto_k_prev'] > 80) & (df['sto_k'] <= 80)

        # Rolling for swing detection
        df['rolling_low'] = df['low'].rolling(window=self.lookback).min()
        df['rolling_high'] = df['high'].rolling(window=self.lookback).max()

        return df

    def backfill_history(self, symbol: str, df: pd.DataFrame):
        logger.info(f"  Backfilling history for {symbol}...")

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
                    'symbol': symbol
                }

                if self.last_signal_type[symbol] != 'ovs':
                    self.ovs_history[symbol].append(signal)
                    self.last_signal_type[symbol] = 'ovs'
                    signals_found += 1
                else:
                    if self.ovs_history[symbol] and signal['value'] < self.ovs_history[symbol][-1]['value']:
                        self.ovs_history[symbol][-1] = signal

            elif bar['overbought']:
                signal = {
                    'datetime': bar_time,
                    'type': 'ovb',
                    'value': bar['rolling_high'],
                    'close': bar['close'],
                    'symbol': symbol
                }

                if self.last_signal_type[symbol] != 'ovb':
                    self.ovb_history[symbol].append(signal)
                    self.last_signal_type[symbol] = 'ovb'
                    signals_found += 1
                else:
                    if self.ovb_history[symbol] and signal['value'] > self.ovb_history[symbol][-1]['value']:
                        self.ovb_history[symbol][-1] = signal

        logger.info(f"  {symbol}: Found {signals_found} signal groups")
        logger.info(f"  {symbol}: OVS: {len(self.ovs_history[symbol])}, OVB: {len(self.ovb_history[symbol])}")

    def check_signal(self, symbol: str, allowed_direction: str) -> Optional[Dict]:
        """
        Check signal for symbol, filtered by allowed direction.
        allowed_direction: 'BUY' or 'SELL'
        """
        # Initialize if needed
        if symbol not in self.ovs_history:
            self.ovs_history[symbol] = []
            self.ovb_history[symbol] = []
            self.last_signal_type[symbol] = None
            self.initialized[symbol] = False
            self.last_processed_time[symbol] = None

        # Get data
        df = self.get_historical_data(symbol)
        if df is None or len(df) < 100:
            return None

        df = self.calculate_indicators(df)

        # Backfill on first run
        if not self.initialized.get(symbol, False):
            self.backfill_history(symbol, df)
            self.initialized[symbol] = True
            self.last_processed_time[symbol] = df.index[-2]
            return None

        latest = df.iloc[-2]
        current_time = df.index[-2]

        if self.last_processed_time[symbol] and current_time <= self.last_processed_time[symbol]:
            return None

        self.last_processed_time[symbol] = current_time

        # Check OVS (for BUY signals) - only if allowed
        if latest['oversold']:
            signal = {
                'datetime': current_time,
                'type': 'ovs',
                'value': latest['rolling_low'],
                'close': latest['close'],
                'symbol': symbol
            }

            if self.last_signal_type[symbol] != 'ovs':
                self.ovs_history[symbol].append(signal)
                self.last_signal_type[symbol] = 'ovs'

                # Check Higher Low -> BUY (only if BUY is allowed)
                if allowed_direction == 'BUY' and len(self.ovs_history[symbol]) >= 2:
                    current = self.ovs_history[symbol][-1]
                    prev = self.ovs_history[symbol][-2]

                    if current['value'] > prev['value']:
                        logger.info(f"  {symbol}: Higher Low! {prev['value']:.2f} -> {current['value']:.2f}")
                        return {
                            'type': 'BUY',
                            'symbol': symbol,
                            'entry': current['close'],
                            'sl': prev['value'],
                            'signal_time': current_time
                        }
            else:
                if self.ovs_history[symbol] and signal['value'] < self.ovs_history[symbol][-1]['value']:
                    self.ovs_history[symbol][-1] = signal

        # Check OVB (for SELL signals) - only if allowed
        elif latest['overbought']:
            signal = {
                'datetime': current_time,
                'type': 'ovb',
                'value': latest['rolling_high'],
                'close': latest['close'],
                'symbol': symbol
            }

            if self.last_signal_type[symbol] != 'ovb':
                self.ovb_history[symbol].append(signal)
                self.last_signal_type[symbol] = 'ovb'

                # Check Lower High -> SELL (only if SELL is allowed)
                if allowed_direction == 'SELL' and len(self.ovb_history[symbol]) >= 2:
                    current = self.ovb_history[symbol][-1]
                    prev = self.ovb_history[symbol][-2]

                    if current['value'] < prev['value']:
                        logger.info(f"  {symbol}: Lower High! {prev['value']:.2f} -> {current['value']:.2f}")
                        return {
                            'type': 'SELL',
                            'symbol': symbol,
                            'entry': current['close'],
                            'sl': prev['value'],
                            'signal_time': current_time
                        }
            else:
                if self.ovb_history[symbol] and signal['value'] > self.ovb_history[symbol][-1]['value']:
                    self.ovb_history[symbol][-1] = signal

        return None

# ============================================================================
# ORDER MANAGEMENT
# ============================================================================

class OrderManager:
    def __init__(self, magic_number: int, risk_percent: float):
        self.magic_number = magic_number
        self.risk_percent = risk_percent

    def get_symbol_info(self, symbol: str) -> Optional[mt5.SymbolInfo]:
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None
        return info

    def calculate_lot_size(self, symbol: str, order_type: str, entry_price: float,
                          sl_price: float, account_balance: float) -> tuple:
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return 0.0, 0.0, False

        risk_amount = account_balance * (self.risk_percent / 100)

        if order_type == 'BUY':
            action = mt5.ORDER_TYPE_BUY
        else:
            action = mt5.ORDER_TYPE_SELL

        profit_1lot = mt5.order_calc_profit(action, symbol, 1.0, entry_price, sl_price)

        if profit_1lot is None:
            logger.error(f"  order_calc_profit failed for {symbol}")
            return symbol_info.volume_min, 0.0, False

        loss_per_lot = abs(profit_1lot)

        if loss_per_lot > 0:
            lots = risk_amount / loss_per_lot
        else:
            lots = symbol_info.volume_min

        lot_step = symbol_info.volume_step
        lots = round(lots / lot_step) * lot_step
        lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))

        expected_profit = mt5.order_calc_profit(action, symbol, lots, entry_price, sl_price)
        expected_loss = abs(expected_profit) if expected_profit else 0

        is_valid = expected_loss <= risk_amount * 2

        logger.info(f"  Lot: {lots} | Risk: ${risk_amount:.2f} | Expected Loss: ${expected_loss:.2f}")

        return lots, expected_loss, is_valid

    def has_position(self, symbol: str) -> bool:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                if pos.magic == self.magic_number:
                    return True
        return False

    def open_position(self, symbol: str, order_type: str, entry: float, sl: float,
                      rr_ratio: float, account_balance: float) -> bool:
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return False

        if order_type == 'BUY':
            risk_points = entry - sl
            tp = entry + (risk_points * rr_ratio)
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        else:
            risk_points = sl - entry
            tp = entry - (risk_points * rr_ratio)
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid

        lots, expected_loss, is_valid = self.calculate_lot_size(
            symbol, order_type, entry, sl, account_balance
        )

        if lots <= 0 or not is_valid:
            logger.error(f"TRADE SKIPPED: {symbol} - Invalid lot or SL too wide")
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type_mt5,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": f"SemiAuto_{order_type}",
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

        logger.info(f"ORDER OPENED: {order_type} {symbol}")
        logger.info(f"  Lots: {lots:.2f} | Entry: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")

        return True

    def get_all_positions(self) -> List:
        all_positions = mt5.positions_get()
        if all_positions is None:
            return []
        return [pos for pos in all_positions if pos.magic == self.magic_number]

# ============================================================================
# MAIN BOT
# ============================================================================

class SemiAutoBot:
    def __init__(self):
        self.mt5_conn = MT5Connection(MT5_CONFIG)
        self.direction_manager = DirectionManager(MAGIC_NUMBER)
        self.signal_detector = SignalDetector(LOOKBACK)
        self.order_manager = OrderManager(MAGIC_NUMBER, RISK_PERCENT)
        self.running = False
        self.symbols = []

    def start(self):
        logger.info("=" * 60)
        logger.info("SEMI-AUTO TRADING BOT - M15")
        logger.info("=" * 60)
        logger.info("Mode: Direction controlled by PENDING ORDERS")
        logger.info("  - Pending BUY  -> Bot takes BUY signals only")
        logger.info("  - Pending SELL -> Bot takes SELL signals only")
        logger.info("  - No pending   -> No trading for that symbol")
        logger.info("=" * 60)

        if not self.mt5_conn.connect():
            logger.error("Failed to connect to MT5. Exiting.")
            return

        # Verify symbols
        valid_symbols = []
        for symbol in SYMBOL_CONFIG.keys():
            info = mt5.symbol_info(symbol)
            if info is not None:
                valid_symbols.append(symbol)
                logger.info(f"[OK] {symbol}")
            else:
                logger.warning(f"[X] {symbol} not found")

        if not valid_symbols:
            logger.error("No valid symbols. Exiting.")
            self.mt5_conn.disconnect()
            return

        self.symbols = valid_symbols
        self.running = True

        # Show initial directions
        self.print_directions()

        logger.info("\nBot is running. Press Ctrl+C to stop.\n")

        try:
            self.main_loop()
        except KeyboardInterrupt:
            logger.info("\nStopping bot...")
        finally:
            self.stop()

    def print_directions(self):
        """Print current allowed directions from pending orders"""
        logger.info("\n--- CURRENT DIRECTIONS (from pending orders) ---")
        for symbol in self.symbols:
            direction = self.direction_manager.get_allowed_direction(symbol)
            if direction:
                logger.info(f"  {symbol}: {direction} signals allowed")
            else:
                logger.info(f"  {symbol}: NO PENDING - not trading")
        logger.info("-" * 50)

    def main_loop(self):
        last_direction_check = datetime.now() - timedelta(minutes=5)

        while self.running:
            try:
                current_time = datetime.now()

                # Print directions every 5 minutes
                if (current_time - last_direction_check).seconds >= 300:
                    self.print_directions()
                    last_direction_check = current_time

                # Check each symbol
                for symbol in self.symbols:
                    # Get allowed direction from pending orders
                    allowed_direction = self.direction_manager.get_allowed_direction(symbol)

                    if allowed_direction is None:
                        continue  # No pending = no trading

                    # Skip if already have position
                    if self.order_manager.has_position(symbol):
                        continue

                    # Check for signal (filtered by allowed direction)
                    signal = self.signal_detector.check_signal(symbol, allowed_direction)

                    if signal:
                        logger.info(f"\n{'=' * 50}")
                        logger.info(f"SIGNAL: {signal['type']} {symbol}")
                        logger.info(f"Direction allowed: {allowed_direction}")
                        logger.info(f"Entry: {signal['entry']:.5f} | SL: {signal['sl']:.5f}")
                        logger.info(f"{'=' * 50}")

                        # Get RR for this symbol
                        rr = SYMBOL_CONFIG.get(symbol, {}).get('rr', 1.5)

                        success = self.order_manager.open_position(
                            symbol=symbol,
                            order_type=signal['type'],
                            entry=signal['entry'],
                            sl=signal['sl'],
                            rr_ratio=rr,
                            account_balance=ACCOUNT_SIZE
                        )

                        if success:
                            logger.info(f"Position opened for {symbol}")
                        else:
                            logger.error(f"Failed to open position for {symbol}")

                # Status every 5 minutes
                if current_time.minute % 5 == 0 and current_time.second < CHECK_INTERVAL:
                    self.print_status()

                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)

    def print_status(self):
        positions = self.order_manager.get_all_positions()
        account = mt5.account_info()

        logger.info(f"\n--- STATUS ({datetime.now().strftime('%H:%M:%S')}) ---")
        logger.info(f"Balance: ${account.balance:,.2f} | Equity: ${account.equity:,.2f}")
        logger.info(f"Open Positions: {len(positions)}")

        for pos in positions:
            profit_str = f"+${pos.profit:.2f}" if pos.profit >= 0 else f"-${abs(pos.profit):.2f}"
            direction = "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT"
            logger.info(f"  {pos.symbol}: {direction} {pos.volume} lots | PnL: {profit_str}")

        # Show pending order directions
        logger.info("Directions:")
        for symbol in self.symbols:
            d = self.direction_manager.get_allowed_direction(symbol)
            logger.info(f"  {symbol}: {d if d else 'NONE'}")

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
    SEMI-AUTO TRADING BOT - M15
    ================================================================
    How to use:
    1. Set PENDING ORDER (BUY LIMIT/STOP or SELL LIMIT/STOP)
       for the symbol you want to trade
    2. Pending BUY  = Bot will only take BUY signals
       Pending SELL = Bot will only take SELL signals
    3. No pending   = Bot will NOT trade that symbol
    4. Pending order is NOT deleted - it's just a direction marker
    ================================================================
    """)

    bot = SemiAutoBot()
    bot.start()
