"""
MT5 Live Trading Bot - RSI + Stochastic Higher Low / Lower High Strategy
Parallel Mode: เทรดทุก symbol พร้อมกัน (แต่ละ symbol มี position ได้ 1 ตัว)
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
# CONFIGURATION - แก้ไขตรงนี้
# ============================================================================

MT5_CONFIG = {
    'login': 1512365884,           # ใส่ Login ID ของคุณ
    'password': '74!!44N77ufd',  # ใส่ Password ของคุณ
    'server': 'FTMO-Demo', # ใส่ Server name (เช่น 'ICMarkets-Demo', 'Exness-MT5Real')
    'path':r"C:\Program Files\MetaTrader 5\terminal64.exe",
}

# Trading Settings
ACCOUNT_SIZE = 10000     # ขนาดบัญชี (ปรับตาม balance จริง)
RISK_PERCENT = 0.5        # Risk 0.5% per trade
LOOKBACK = 10             # Rolling window for swing detection
MAGIC_NUMBER = 20250121   # Magic number สำหรับระบุ orders ของ bot นี้

# Optimal RR per symbol (จาก optimization)
OPTIMAL_RR = {
    'XAUUSD': 1.1,
    'US30.cash': 1.6,
    'UKOIL.cash': 1.6,      # Brent Oil
    'US100.cash': 1.0,       # NQ Futures (ชื่ออาจต่างกันแต่ละ broker)
}

# Symbol mapping (MT5 symbol name -> display name)
# ปรับตาม broker ของคุณ
SYMBOLS = {
    'XAUUSD': {'name': 'XAUUSD', 'rr': 1.1},
    'US30.cash': {'name': 'US30.cash', 'rr': 1.6},
    'UKOIL.cash': {'name': 'Brent', 'rr': 1.6},       # หรือ 'UKOIL', 'BRENT' แล้วแต่ broker
    'US100.cash': {'name': 'NQ Futures', 'rr': 1.0},   # หรือ 'USTEC', 'NAS100' แล้วแต่ broker
}

# Check interval (seconds)
CHECK_INTERVAL = 60  # ตรวจสอบทุก 1 นาที (1H candle = 3600 แต่เราเช็คถี่กว่าเพื่อไม่พลาด)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('mt5_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MT5 CONNECTION
# ============================================================================

class MT5Connection:
    """จัดการการเชื่อมต่อ MT5"""

    def __init__(self, config: dict):
        self.config = config
        self.connected = False

    def connect(self) -> bool:
        """เชื่อมต่อ MT5"""
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
        logger.info(f"Server: {account_info.server}")

        self.connected = True
        return True

    def disconnect(self):
        """ตัดการเชื่อมต่อ MT5"""
        mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")

    def get_account_balance(self) -> float:
        """ดึง balance ปัจจุบัน"""
        account_info = mt5.account_info()
        return account_info.balance if account_info else 0

# ============================================================================
# SIGNAL DETECTION
# ============================================================================

class SignalDetector:
    """ตรวจจับ signals จาก RSI + Stochastic"""

    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.ovs_history: Dict[str, List] = {}  # Oversold signals history
        self.ovb_history: Dict[str, List] = {}  # Overbought signals history
        self.last_signal_type: Dict[str, str] = {}  # Track last signal type per symbol
        self.initialized: Dict[str, bool] = {}  # Track if symbol history is initialized
        self.last_processed_time: Dict[str, datetime] = {}  # Track last processed candle time

    def get_historical_data(self, symbol: str, timeframe=mt5.TIMEFRAME_H1, bars: int = 500) -> Optional[pd.DataFrame]:
        """ดึงข้อมูลราคาจาก MT5"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

        if rates is None or len(rates) == 0:
            logger.warning(f"Failed to get data for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """คำนวณ Stochastic"""
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

        # Previous stochastic for crossover detection
        df['sto_k_prev'] = df['sto_k'].shift(1)

        # Oversold crossover: K crosses above 20
        df['oversold'] = (df['sto_k_prev'] < 20) & (df['sto_k'] >= 20)

        # Overbought crossover: K crosses below 80
        df['overbought'] = (df['sto_k_prev'] > 80) & (df['sto_k'] <= 80)

        # Rolling low/high for swing detection
        df['rolling_low'] = df['low'].rolling(window=self.lookback).min()
        df['rolling_high'] = df['high'].rolling(window=self.lookback).max()

        return df

    def backfill_history(self, symbol: str, df: pd.DataFrame):
        """
        ดึงข้อมูลย้อนหลังและสร้าง signal history
        ใช้ตอนเริ่มต้น bot เพื่อให้มี context ก่อนหน้า

        Logic (ตรงกับ backtest):
        - ถ้า signal ชนิดเดียวกันมาติดๆ กัน -> UPDATE เป็นอันที่ดีที่สุด
        - OVS: เก็บอันที่ rolling_low ต่ำสุด
        - OVB: เก็บอันที่ rolling_high สูงสุด
        - เมื่อ signal สลับชนิด -> เริ่ม group ใหม่
        """
        logger.info(f"  Backfilling history for {symbol}...")

        # ใช้ทุกแท่งยกเว้นแท่งสุดท้าย (แท่งที่กำลังสร้างอยู่)
        history_bars = df.iloc[:-1]

        logger.info(f"  {symbol}: Processing {len(history_bars)} bars from {history_bars.index[0]} to {history_bars.index[-1]}")

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
                    # Signal ชนิดใหม่ -> เพิ่มเข้า history
                    self.ovs_history[symbol].append(signal)
                    self.last_signal_type[symbol] = 'ovs'
                    signals_found += 1
                else:
                    # Signal ชนิดเดียวกัน -> UPDATE ถ้าค่าดีกว่า (OVS = ต่ำกว่า)
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
                    # Signal ชนิดใหม่ -> เพิ่มเข้า history
                    self.ovb_history[symbol].append(signal)
                    self.last_signal_type[symbol] = 'ovb'
                    signals_found += 1
                else:
                    # Signal ชนิดเดียวกัน -> UPDATE ถ้าค่าดีกว่า (OVB = สูงกว่า)
                    if self.ovb_history[symbol] and signal['value'] > self.ovb_history[symbol][-1]['value']:
                        self.ovb_history[symbol][-1] = signal

        logger.info(f"  {symbol}: Found {signals_found} signal groups (alternating)")
        logger.info(f"  {symbol}: OVS history: {len(self.ovs_history[symbol])}, OVB history: {len(self.ovb_history[symbol])}")

        # Log last signals for debugging
        if self.ovs_history[symbol]:
            last_ovs = self.ovs_history[symbol][-1]
            logger.info(f"  {symbol}: Last OVS at {last_ovs['datetime']}, value={last_ovs['value']:.5f}")
        if self.ovb_history[symbol]:
            last_ovb = self.ovb_history[symbol][-1]
            logger.info(f"  {symbol}: Last OVB at {last_ovb['datetime']}, value={last_ovb['value']:.5f}")

    def check_signal(self, symbol: str) -> Optional[Dict]:
        """
        ตรวจสอบ signal สำหรับ symbol
        Returns: dict with signal info or None
        """
        # Initialize history if needed
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

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Backfill history on first run
        if not self.initialized.get(symbol, False):
            self.backfill_history(symbol, df)
            self.initialized[symbol] = True
            self.last_processed_time[symbol] = df.index[-2]
            return None  # Don't trade on first run, just initialize

        # Get latest complete bar (not current forming bar)
        latest = df.iloc[-2]  # -2 = last complete bar, -1 = current forming
        current_time = df.index[-2]

        # Skip if we already processed this candle
        if self.last_processed_time[symbol] and current_time <= self.last_processed_time[symbol]:
            return None

        self.last_processed_time[symbol] = current_time

        # Check for new oversold signal
        if latest['oversold']:
            signal = {
                'datetime': current_time,
                'type': 'ovs',
                'value': latest['rolling_low'],
                'close': latest['close'],
                'symbol': symbol
            }

            if self.last_signal_type[symbol] != 'ovs':
                # Signal ชนิดใหม่ -> เพิ่มเข้า history และเช็ค Higher Low
                self.ovs_history[symbol].append(signal)
                self.last_signal_type[symbol] = 'ovs'

                # Check Higher Low condition
                if len(self.ovs_history[symbol]) >= 2:
                    current = self.ovs_history[symbol][-1]
                    prev = self.ovs_history[symbol][-2]

                    if current['value'] > prev['value']:
                        logger.info(f"  {symbol}: Higher Low detected! {prev['value']:.2f} -> {current['value']:.2f}")
                        return {
                            'type': 'BUY',
                            'symbol': symbol,
                            'entry': current['close'],
                            'sl': prev['value'],
                            'signal_time': current_time
                        }
            else:
                # Signal ชนิดเดียวกัน -> UPDATE ถ้าค่าดีกว่า (OVS = ต่ำกว่า)
                if self.ovs_history[symbol] and signal['value'] < self.ovs_history[symbol][-1]['value']:
                    logger.info(f"  {symbol}: Updating OVS to better value: {signal['value']:.2f}")
                    self.ovs_history[symbol][-1] = signal

        # Check for new overbought signal
        elif latest['overbought']:
            signal = {
                'datetime': current_time,
                'type': 'ovb',
                'value': latest['rolling_high'],
                'close': latest['close'],
                'symbol': symbol
            }

            if self.last_signal_type[symbol] != 'ovb':
                # Signal ชนิดใหม่ -> เพิ่มเข้า history และเช็ค Lower High
                self.ovb_history[symbol].append(signal)
                self.last_signal_type[symbol] = 'ovb'

                # Check Lower High condition
                if len(self.ovb_history[symbol]) >= 2:
                    current = self.ovb_history[symbol][-1]
                    prev = self.ovb_history[symbol][-2]

                    if current['value'] < prev['value']:
                        logger.info(f"  {symbol}: Lower High detected! {prev['value']:.2f} -> {current['value']:.2f}")
                        return {
                            'type': 'SELL',
                            'symbol': symbol,
                            'entry': current['close'],
                            'sl': prev['value'],
                            'signal_time': current_time
                        }
            else:
                # Signal ชนิดเดียวกัน -> UPDATE ถ้าค่าดีกว่า (OVB = สูงกว่า)
                if self.ovb_history[symbol] and signal['value'] > self.ovb_history[symbol][-1]['value']:
                    logger.info(f"  {symbol}: Updating OVB to better value: {signal['value']:.2f}")
                    self.ovb_history[symbol][-1] = signal

        return None

# ============================================================================
# ORDER MANAGEMENT
# ============================================================================

class OrderManager:
    """จัดการ Orders และ Positions"""

    def __init__(self, magic_number: int, risk_percent: float):
        self.magic_number = magic_number
        self.risk_percent = risk_percent

    def get_symbol_info(self, symbol: str) -> Optional[mt5.SymbolInfo]:
        """ดึงข้อมูล symbol"""
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
        """
        Calculate lot size using mt5.order_calc_profit() for accuracy

        Returns: (lot_size, expected_loss, is_valid)
        """
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return 0.0, 0.0, False

        risk_amount = account_balance * (self.risk_percent / 100)

        # Determine action type
        if order_type == 'BUY':
            action = mt5.ORDER_TYPE_BUY
        else:
            action = mt5.ORDER_TYPE_SELL

        # Calculate loss per 1 lot using MT5's function
        profit_1lot = mt5.order_calc_profit(action, symbol, 1.0, entry_price, sl_price)

        if profit_1lot is None:
            logger.error(f"  order_calc_profit failed for {symbol}")
            return symbol_info.volume_min, 0.0, False

        loss_per_lot = abs(profit_1lot)

        logger.info(f"  Lot Calc: {symbol} {order_type}")
        logger.info(f"    Entry: {entry_price:.2f}, SL: {sl_price:.2f}")
        logger.info(f"    Loss per 1 lot: ${loss_per_lot:.2f}")
        logger.info(f"    Target Risk: ${risk_amount:.2f}")

        # Calculate required lot
        if loss_per_lot > 0:
            lots = risk_amount / loss_per_lot
        else:
            lots = symbol_info.volume_min

        # Round to lot step
        lot_step = symbol_info.volume_step
        lots = round(lots / lot_step) * lot_step

        # Clamp to min/max
        lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))

        # Calculate expected loss with this lot size
        expected_profit = mt5.order_calc_profit(action, symbol, lots, entry_price, sl_price)
        expected_loss = abs(expected_profit) if expected_profit else 0

        logger.info(f"    Calculated Lot: {lots}")
        logger.info(f"    Expected Loss: ${expected_loss:.2f}")

        # Check if expected loss is acceptable (within 2x of risk)
        is_valid = expected_loss <= risk_amount * 2

        if not is_valid:
            logger.warning(f"    WARNING: Expected loss ${expected_loss:.2f} > 2x Risk ${risk_amount:.2f}")
            logger.warning(f"    Trade will be SKIPPED - SL too wide for account size!")

        return lots, expected_loss, is_valid

    def has_position(self, symbol: str) -> bool:
        """ตรวจสอบว่ามี position อยู่หรือไม่"""
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                if pos.magic == self.magic_number:
                    return True
        return False

    def get_position(self, symbol: str) -> Optional[mt5.TradePosition]:
        """ดึง position ของ symbol"""
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                if pos.magic == self.magic_number:
                    return pos
        return None

    def open_position(self, symbol: str, order_type: str, entry: float, sl: float,
                      rr_ratio: float, account_balance: float) -> bool:
        """Open new position"""

        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return False

        # Calculate TP based on RR
        if order_type == 'BUY':
            risk_points = entry - sl
            tp = entry + (risk_points * rr_ratio)
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        else:  # SELL
            risk_points = sl - entry
            tp = entry - (risk_points * rr_ratio)
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid

        # Calculate lot size using accurate MT5 function
        lots, expected_loss, is_valid = self.calculate_lot_size(
            symbol, order_type, entry, sl, account_balance
        )

        if lots <= 0:
            logger.error(f"Invalid lot size calculated for {symbol}")
            return False

        # Skip trade if expected loss is too high
        if not is_valid:
            logger.error(f"TRADE SKIPPED: {symbol} - Expected loss ${expected_loss:.2f} too high!")
            logger.error(f"  Consider using smaller SL or larger account size")
            return False

        # Prepare request
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
            "comment": f"RSI_STO_Bot_{order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Order send failed for {symbol}: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed for {symbol}: {result.comment} (code: {result.retcode})")
            return False

        logger.info(f"ORDER OPENED: {order_type} {symbol}")
        logger.info(f"  Lots: {lots:.2f} | Entry: {price:.5f}")
        logger.info(f"  SL: {sl:.5f} | TP: {tp:.5f} | RR: 1:{rr_ratio}")

        return True

    def get_all_positions(self) -> List[mt5.TradePosition]:
        """ดึง positions ทั้งหมดของ bot นี้"""
        all_positions = mt5.positions_get()
        if all_positions is None:
            return []
        return [pos for pos in all_positions if pos.magic == self.magic_number]

# ============================================================================
# MAIN BOT
# ============================================================================

class TradingBot:
    """Main Trading Bot"""

    def __init__(self):
        self.mt5_conn = MT5Connection(MT5_CONFIG)
        self.signal_detector = SignalDetector(LOOKBACK)
        self.order_manager = OrderManager(MAGIC_NUMBER, RISK_PERCENT)
        self.running = False
        self.last_check_time = {}

    def start(self):
        """เริ่มการทำงานของ bot"""
        logger.info("="*60)
        logger.info("RSI + STOCHASTIC TRADING BOT - STARTING")
        logger.info("="*60)
        logger.info(f"Risk: {RISK_PERCENT}% per trade")
        logger.info(f"Symbols: {list(SYMBOLS.keys())}")
        logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
        logger.info("="*60)

        # Connect to MT5
        if not self.mt5_conn.connect():
            logger.error("Failed to connect to MT5. Exiting.")
            return

        # Verify symbols exist
        valid_symbols = {}
        for symbol, config in SYMBOLS.items():
            info = mt5.symbol_info(symbol)
            if info is not None:
                valid_symbols[symbol] = config
                logger.info(f"[OK] {symbol} found")
            else:
                logger.warning(f"[X] {symbol} not found - skipping")

        if not valid_symbols:
            logger.error("No valid symbols found. Exiting.")
            self.mt5_conn.disconnect()
            return

        self.symbols = valid_symbols
        self.running = True

        logger.info("\nBot is now running. Press Ctrl+C to stop.\n")

        try:
            self.main_loop()
        except KeyboardInterrupt:
            logger.info("\nStopping bot...")
        finally:
            self.stop()

    def main_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                current_time = datetime.now()

                # Check each symbol
                for symbol, config in self.symbols.items():
                    # Skip if already have position for this symbol
                    if self.order_manager.has_position(symbol):
                        continue

                    # Check for signal
                    signal = self.signal_detector.check_signal(symbol)

                    if signal:
                        logger.info(f"\n{'='*40}")
                        logger.info(f"SIGNAL DETECTED: {signal['type']} {symbol}")
                        logger.info(f"Entry: {signal['entry']:.5f} | SL: {signal['sl']:.5f}")
                        logger.info(f"{'='*40}")

                        # Open position - ใช้ ACCOUNT_SIZE คงที่แทน balance จริง
                        success = self.order_manager.open_position(
                            symbol=symbol,
                            order_type=signal['type'],
                            entry=signal['entry'],
                            sl=signal['sl'],
                            rr_ratio=config['rr'],
                            account_balance=ACCOUNT_SIZE  # Fixed account size
                        )

                        if success:
                            logger.info(f"Position opened successfully for {symbol}")
                        else:
                            logger.error(f"Failed to open position for {symbol}")

                # Show status every 5 minutes
                if current_time.minute % 5 == 0 and current_time.second < CHECK_INTERVAL:
                    self.print_status()

                # Wait before next check
                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)

    def print_status(self):
        """แสดงสถานะปัจจุบัน"""
        positions = self.order_manager.get_all_positions()
        account = mt5.account_info()

        logger.info(f"\n--- STATUS ({datetime.now().strftime('%H:%M:%S')}) ---")
        logger.info(f"Balance: ${account.balance:,.2f} | Equity: ${account.equity:,.2f}")
        logger.info(f"Open Positions: {len(positions)}")

        for pos in positions:
            profit_str = f"+${pos.profit:.2f}" if pos.profit >= 0 else f"-${abs(pos.profit):.2f}"
            direction = "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT"
            logger.info(f"  {pos.symbol}: {direction} {pos.volume} lots | PnL: {profit_str}")

        logger.info("-" * 40)

    def stop(self):
        """หยุดการทำงานของ bot"""
        self.running = False
        self.mt5_conn.disconnect()
        logger.info("Bot stopped.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("""
    ================================================================
    RSI + STOCHASTIC TRADING BOT FOR MT5
    Strategy: Higher Low / Lower High
    Mode: Parallel (all symbols trade independently)
    ================================================================
    """)

    # Check if config is set
    if MT5_CONFIG['login'] == 123456789:
        print("ERROR: Please edit MT5_CONFIG with your account details!")
        print("Open mt5_live_bot.py and update:")
        print("  - login")
        print("  - password")
        print("  - server")
        exit(1)

    # Start bot
    bot = TradingBot()
    bot.start()
