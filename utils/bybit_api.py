import time
import hmac
import hashlib
import json
import logging
import requests
from typing import Dict, List, Optional, Tuple, Union
import websocket
import threading
import ccxt
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)

class BybitAPI:
    """Wrapper for Bybit API interactions"""
    
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        """
        Initialize Bybit API connection
        
        Parameters:
        api_key (str): Bybit API key
        api_secret (str): Bybit API secret
        testnet (bool): Use testnet if True, mainnet if False
        """
        self.api_key = api_key or config.BYBIT_API_KEY
        self.api_secret = api_secret or config.BYBIT_API_SECRET
        self.testnet = testnet if testnet is not None else config.BYBIT_TESTNET
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret are required for Bybit API")
        
        # Set up CCXT client for easier API interaction
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        
        # Set to testnet if configured
        if self.testnet:
            self.exchange.urls['api'] = self.exchange.urls['test']
        
        # Request counters for rate limiting
        self.request_count = 0
        self.last_request_timestamp = time.time()
        self.request_weights = {}
        
        # WebSocket connections
        self.ws_connections = {}
        self.ws_callbacks = {}
        self.ws_authenticated = {}
        self.ws_stop_event = threading.Event()
    
    def _check_rate_limits(self):
        """Check and handle rate limits"""
        current_time = time.time()
        time_elapsed = current_time - self.last_request_timestamp
        
        # Reset counters if a minute has passed
        if time_elapsed >= config.REQUEST_WEIGHT_RESET:
            self.request_count = 0
            self.last_request_timestamp = current_time
        
        # If approaching rate limit, sleep for the remaining time
        if self.request_count >= config.MAX_REQUESTS_PER_MINUTE:
            sleep_time = config.REQUEST_WEIGHT_RESET - time_elapsed
            if sleep_time > 0:
                logger.info(f"Rate limit approached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_timestamp = time.time()
    
    def _update_request_count(self, weight=1):
        """Update the request counter"""
        self.request_count += weight
    
    def get_server_time(self) -> int:
        """Get Bybit server time"""
        self._check_rate_limits()
        result = self.exchange.public_get_v5_market_time()
        self._update_request_count()
        
        return int(result['result']['timeNano']) // 1_000_000
    
    def get_account_balance(self, coin="USDT") -> Dict:
        """
        Get account balance
        
        Parameters:
        coin (str): The coin to get balance for (default: USDT)
        
        Returns:
        dict: Account balance information
        """
        self._check_rate_limits()
        balances = self.exchange.fetch_balance()
        self._update_request_count()
        
        if coin in balances['total']:
            return {
                'coin': coin,
                'free': balances['free'].get(coin, 0),
                'used': balances['used'].get(coin, 0),
                'total': balances['total'].get(coin, 0)
            }
        else:
            return {
                'coin': coin,
                'free': 0,
                'used': 0,
                'total': 0
            }
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200, 
                  start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict]:
        """
        Get historical k-line data
        
        Parameters:
        symbol (str): Trading pair symbol
        interval (str): Time interval (1m, 5m, 1h, 1D, etc.)
        limit (int): Number of candles to return (max 1000)
        start_time (int): Start timestamp in milliseconds
        end_time (int): End timestamp in milliseconds
        
        Returns:
        List[Dict]: List of candlestick data
        """
        self._check_rate_limits()
        
        # Map interval to CCXT format if needed
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '1h': '1h',
            '1D': '1d',
        }
        ccxt_interval = interval_map.get(interval, interval)
        
       # Get candles
        params = {}
        if start_time:
            params['since'] = start_time
        if end_time:
            params['until'] = end_time

        params['recvWindow'] = 20000  # 20 seconds, increase if needed
        params['recv_window'] = 20000
            
        try:
            candles = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=ccxt_interval,
                limit=limit,
                params=params
            )
            self._update_request_count()
            
            # Convert to consistent format
            formatted_candles = []
            for candle in candles:
                formatted_candles.append({
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                })
            
            return formatted_candles
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return []
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker information
        
        Parameters:
        symbol (str): Trading pair symbol
        
        Returns:
        Dict: Ticker information
        """
        self._check_rate_limits()
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            self._update_request_count()
            
            return {
                'symbol': symbol,
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'volume': ticker['volume'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}
    
    def get_leverage_brackets(self, symbol: str) -> List[Dict]:
        """
        Get leverage brackets for a symbol
        
        Parameters:
        symbol (str): Trading pair symbol
        
        Returns:
        List[Dict]: Leverage brackets
        """
        self._check_rate_limits()
        try:
            response = self.exchange.fetch_derivatives_market_leverage_tiers(symbol)
            self._update_request_count()
            
            if not response or 'info' not in response or 'tiers' not in response['info']:
                return []
                
            return response['info']['tiers']
        except Exception as e:
            logger.error(f"Error fetching leverage brackets for {symbol}: {e}")
            return []
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol
        
        Parameters:
        symbol (str): Trading pair symbol
        leverage (int): Leverage value (1-125)
        
        Returns:
        bool: Success or failure
        """
        self._check_rate_limits()
        try:
            response = self.exchange.set_leverage(leverage, symbol)
            self._update_request_count()
            
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            return False
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get current positions
        
        Parameters:
        symbol (str, optional): Trading pair symbol, None for all positions
        
        Returns:
        List[Dict]: List of positions
        """
        self._check_rate_limits()
        try:
            if symbol:
                positions = self.exchange.fetch_positions([symbol])
            else:
                positions = self.exchange.fetch_positions()
                
            self._update_request_count()
            
            # Format positions consistently
            formatted_positions = []
            for pos in positions:
                if pos['contracts'] > 0:  # Only include open positions
                    formatted_positions.append({
                        'symbol': pos['symbol'],
                        'size': pos['contracts'],
                        'side': pos['side'],
                        'entry_price': pos['entryPrice'],
                        'leverage': pos['leverage'],
                        'unrealized_pnl': pos['unrealizedPnl'],
                        'margin_type': pos['marginType'],
                        'liquidation_price': pos['liquidationPrice']
                    })
            
            return formatted_positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, 
                    price: Optional[float] = None, stop_loss: Optional[float] = None, 
                    take_profit: Optional[float] = None, reduce_only: bool = False) -> Dict:
        """
        Create a new order
        
        Parameters:
        symbol (str): Trading pair symbol
        order_type (str): Order type ('limit', 'market')
        side (str): Order side ('buy', 'sell')
        amount (float): Order quantity
        price (float, optional): Order price (required for limit orders)
        stop_loss (float, optional): Stop loss price
        take_profit (float, optional): Take profit price
        reduce_only (bool): If True, the order will only reduce a position, not open a new one
        
        Returns:
        Dict: Order information
        """
        self._check_rate_limits()
        try:
            params = {}
            
            # Add stop loss and take profit if provided
            if stop_loss:
                params['stopLoss'] = stop_loss
            if take_profit:
                params['takeProfit'] = take_profit
            if reduce_only:
                params['reduceOnly'] = True
            
            # Create order based on type
            if order_type.lower() == 'market':
                order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=side.lower(),
                    amount=amount,
                    params=params
                )
            elif order_type.lower() == 'limit':
                if price is None:
                    raise ValueError("Price is required for limit orders")
                order = self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side.lower(),
                    amount=amount,
                    price=price,
                    params=params
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            self._update_request_count(2)  # Order creation has higher weight
            
            logger.info(f"Created {order_type} {side} order for {amount} {symbol}")
            
            return {
                'id': order['id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'price': order['price'],
                'amount': order['amount'],
                'status': order['status'],
                'timestamp': order['timestamp']
            }
        except Exception as e:
            logger.error(f"Error creating order for {symbol}: {e}")
            return {}
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order
        
        Parameters:
        order_id (str): Order ID
        symbol (str): Trading pair symbol
        
        Returns:
        bool: Success or failure
        """
        self._check_rate_limits()
        try:
            self.exchange.cancel_order(order_id, symbol)
            self._update_request_count()
            
            logger.info(f"Cancelled order {order_id} for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order(self, order_id: str, symbol: str) -> Dict:
        """
        Get order information
        
        Parameters:
        order_id (str): Order ID
        symbol (str): Trading pair symbol
        
        Returns:
        Dict: Order information
        """
        self._check_rate_limits()
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            self._update_request_count()
            
            return {
                'id': order['id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'price': order['price'],
                'amount': order['amount'],
                'filled': order['filled'],
                'remaining': order['remaining'],
                'status': order['status'],
                'timestamp': order['timestamp']
            }
        except Exception as e:
            logger.error(f"Error fetching order {order_id}: {e}")
            return {}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open orders
        
        Parameters:
        symbol (str, optional): Trading pair symbol, None for all symbols
        
        Returns:
        List[Dict]: List of open orders
        """
        self._check_rate_limits()
        try:
            if symbol:
                orders = self.exchange.fetch_open_orders(symbol)
            else:
                orders = self.exchange.fetch_open_orders()
                
            self._update_request_count()
            
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'price': order['price'],
                    'amount': order['amount'],
                    'filled': order['filled'],
                    'remaining': order['remaining'],
                    'status': order['status'],
                    'timestamp': order['timestamp']
                })
            
            return formatted_orders
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Get order history
        
        Parameters:
        symbol (str, optional): Trading pair symbol, None for all symbols
        limit (int): Number of orders to return
        
        Returns:
        List[Dict]: List of completed orders
        """
        self._check_rate_limits()
        try:
            if symbol:
                orders = self.exchange.fetch_closed_orders(symbol, limit=limit)
            else:
                orders = self.exchange.fetch_closed_orders(limit=limit)
                
            self._update_request_count()
            
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'price': order['price'],
                    'amount': order['amount'],
                    'filled': order['filled'],
                    'cost': order['cost'],
                    'fee': order['fee'],
                    'status': order['status'],
                    'timestamp': order['timestamp']
                })
            
            return formatted_orders
        except Exception as e:
            logger.error(f"Error fetching order history: {e}")
            return []
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, 
                              risk_amount: float, leverage: int = 1) -> float:
        """
        Calculate position size based on risk parameters
        
        Parameters:
        symbol (str): Trading pair symbol
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        risk_amount (float): Amount to risk in USDT
        leverage (int): Leverage to use
        
        Returns:
        float: Position size in base currency
        """
        try:
            # Calculate risk per contract
            risk_per_contract = abs(entry_price - stop_loss) / entry_price
            
            # Calculate position size
            position_size = (risk_amount / risk_per_contract) * leverage
            
            # Adjust position size based on symbol precision
            markets = self.exchange.load_markets()
            if symbol in markets:
                market = markets[symbol]
                precision = market['precision']['amount']
                position_size = round(position_size, precision)
            
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    # WebSocket methods
    def connect_websocket(self, streams: Union[str, List[str]], callback, private: bool = False):
        """
        Connect to WebSocket stream
        
        Parameters:
        streams (str or List[str]): Stream or list of streams to subscribe to
        callback (function): Callback function for received messages
        private (bool): Whether this is a private stream
        
        Returns:
        str: Connection ID
        """
        if isinstance(streams, str):
            streams = [streams]
            
        base_url = "wss://stream.bybit.com/v5/"
        endpoint = "private" if private else "public"
        url = f"{base_url}{endpoint}"
        
        connection_id = f"{'-'.join(streams)}"
        
        # Store callback
        self.ws_callbacks[connection_id] = callback
        
        # Start WebSocket in a separate thread
        ws_thread = threading.Thread(
            target=self._run_websocket,
            args=(url, streams, connection_id, private)
        )
        ws_thread.daemon = True
        ws_thread.start()
        
        return connection_id
    
    def disconnect_websocket(self, connection_id: str):
        """Disconnect a WebSocket connection"""
        if connection_id in self.ws_connections:
            self.ws_connections[connection_id].close()
            del self.ws_connections[connection_id]
            del self.ws_callbacks[connection_id]
            logger.info(f"Disconnected WebSocket: {connection_id}")
    
    def disconnect_all_websockets(self):
        """Disconnect all WebSocket connections"""
        self.ws_stop_event.set()
        for connection_id in list(self.ws_connections.keys()):
            self.disconnect_websocket(connection_id)
    
    def _run_websocket(self, url: str, streams: List[str], connection_id: str, private: bool):
        """Run a WebSocket connection in a loop"""
        reconnect_count = 0
        
        while not self.ws_stop_event.is_set() and reconnect_count < config.MAX_RECONNECT_ATTEMPTS:
            try:
                # Connect to WebSocket
                ws = websocket.WebSocketApp(
                    url,
                    on_message=lambda ws, msg: self._on_ws_message(connection_id, msg),
                    on_error=lambda ws, err: self._on_ws_error(connection_id, err),
                    on_close=lambda ws, close_status, close_msg: self._on_ws_close(connection_id),
                    on_open=lambda ws: self._on_ws_open(ws, streams, connection_id, private)
                )
                
                self.ws_connections[connection_id] = ws
                
                # Run WebSocket
                ws.run_forever(ping_interval=config.WEBSOCKET_PING_INTERVAL, reconnect=5)
                
                # If we get here, the connection was closed
                if self.ws_stop_event.is_set():
                    break
                    
                # Reconnect after delay
                reconnect_count += 1
                logger.warning(f"WebSocket disconnected, reconnecting ({reconnect_count}/{config.MAX_RECONNECT_ATTEMPTS})")
                time.sleep(config.WEBSOCKET_RECONNECT_DELAY * reconnect_count)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                reconnect_count += 1
                time.sleep(config.WEBSOCKET_RECONNECT_DELAY * reconnect_count)
        
        if reconnect_count >= config.MAX_RECONNECT_ATTEMPTS:
            logger.error(f"Max reconnection attempts reached for {connection_id}")
    
    def _on_ws_open(self, ws, streams: List[str], connection_id: str, private: bool):
        """WebSocket connection opened, send subscription messages"""
        logger.info(f"WebSocket connected: {connection_id}")
        
        try:
            if private:
                # Authenticate for private streams
                expires = int((time.time() + 10) * 1000)
                signature = self._generate_signature("GET", "/realtime_private", expires, "")
                
                auth_message = json.dumps({
                    "op": "auth",
                    "args": [self.api_key, expires, signature]
                })
                ws.send(auth_message)
                self.ws_authenticated[connection_id] = False
            
            # Subscribe to streams
            subscribe_message = json.dumps({
                "op": "subscribe",
                "args": streams
            })
            ws.send(subscribe_message)
            
            logger.info(f"Subscribed to streams: {streams}")
        except Exception as e:
            logger.error(f"Error during WebSocket setup: {e}")
    
    def _on_ws_message(self, connection_id: str, message: str):
        """Handle WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle authentication response for private streams
            if 'op' in data and data['op'] == 'auth':
                if data.get('success'):
                    logger.info("WebSocket authentication successful")
                    self.ws_authenticated[connection_id] = True
                else:
                    logger.error(f"WebSocket authentication failed: {data.get('ret_msg')}")
                return
            
            # Handle subscription success
            if 'op' in data and data['op'] == 'subscribe':
                logger.info(f"Subscription successful: {data}")
                return
            
            # Handle ping/pong
            if 'op' in data and data['op'] == 'ping':
                pong_message = json.dumps({"op": "pong"})
                if connection_id in self.ws_connections:
                    self.ws_connections[connection_id].send(pong_message)
                return
            
            # Process actual data
            if connection_id in self.ws_callbacks:
                self.ws_callbacks[connection_id](data)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in WebSocket message: {message}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_ws_error(self, connection_id: str, error):
        """Handle WebSocket error"""
        logger.error(f"WebSocket error for {connection_id}: {error}")
    
    def _on_ws_close(self, connection_id: str):
        """Handle WebSocket close"""
        logger.info(f"WebSocket closed: {connection_id}")
    
    def _generate_signature(self, method: str, endpoint: str, timestamp: int, data: str) -> str:
        """Generate signature for authentication"""
        param_str = str(timestamp) + self.api_key + method + endpoint + data
        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(param_str, 'utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature