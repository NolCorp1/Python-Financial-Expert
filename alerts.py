#!/usr/bin/env python3
"""
Alert System Module for Double Bottom Pattern Scanner.

This module provides alerting functionality for newly confirmed patterns.
Can be extended to support webhooks, email, or other notification methods.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import json
import pandas as pd


@dataclass
class Alert:
    """Represents an alert for a confirmed pattern."""
    symbol: str
    alert_type: str
    timestamp: datetime
    message: str
    pattern_data: Dict[str, Any]
    priority: str = 'normal'
    sent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'symbol': self.symbol,
            'alert_type': self.alert_type,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'priority': self.priority,
            'sent': self.sent,
            'pattern_data': self.pattern_data
        }
    
    def __repr__(self):
        return f"Alert({self.symbol}, {self.alert_type}, {self.priority})"


class AlertManager:
    """
    Manages pattern alerts and notifications.
    
    Features:
    - Check for new confirmed patterns
    - Send alerts via configurable handlers
    - Track sent alerts to avoid duplicates
    - Support for custom alert handlers (console, webhook, etc.)
    """
    
    def __init__(self):
        """Initialize the AlertManager."""
        self.alerts: List[Alert] = []
        self.sent_alerts: set = set()
        self.handlers: List[Callable[[Alert], None]] = [self._console_handler]
        self.pattern_cache: Dict[str, List[str]] = {}
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """
        Add a custom alert handler.
        
        Args:
            handler: Callable that takes an Alert object
        """
        self.handlers.append(handler)
    
    def _get_pattern_key(self, pattern: Dict) -> str:
        """Generate a unique key for a pattern to avoid duplicate alerts."""
        return f"{pattern.get('symbol', '')}_{pattern.get('bottom1_date', '')}_{pattern.get('bottom2_date', '')}"
    
    def check_for_alerts(self, patterns: List[Dict], 
                         price_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[Alert]:
        """
        Check for new confirmed patterns that should trigger alerts.
        
        An alert is triggered when:
        - Pattern status is 'confirmed' (price broke above neckline)
        - Pattern hasn't been alerted before
        
        Args:
            patterns: List of pattern dictionaries from detection
            price_data: Optional price data for additional checks
            
        Returns:
            List of new Alert objects
        """
        new_alerts = []
        
        for pattern in patterns:
            if pattern.get('status') != 'confirmed':
                continue
            
            pattern_key = self._get_pattern_key(pattern)
            
            if pattern_key in self.sent_alerts:
                continue
            
            symbol = pattern.get('symbol', 'UNKNOWN')
            
            priority = self._determine_priority(pattern)
            
            message = self._create_alert_message(pattern)
            
            alert = Alert(
                symbol=symbol,
                alert_type='BREAKOUT',
                timestamp=datetime.now(),
                message=message,
                pattern_data=pattern,
                priority=priority
            )
            
            new_alerts.append(alert)
            self.alerts.append(alert)
        
        return new_alerts
    
    def _determine_priority(self, pattern: Dict) -> str:
        """Determine alert priority based on pattern strength."""
        strength = pattern.get('strength_score', 0)
        
        if strength >= 80:
            return 'high'
        elif strength >= 60:
            return 'normal'
        else:
            return 'low'
    
    def _create_alert_message(self, pattern: Dict) -> str:
        """Create alert message from pattern data."""
        symbol = pattern.get('symbol', 'UNKNOWN')
        strength = pattern.get('strength_score', 0)
        neckline = pattern.get('neckline', 0)
        target = pattern.get('target_price', 0)
        current_price = pattern.get('current_price', 0)
        
        support = (pattern.get('bottom1_price', 0) + pattern.get('bottom2_price', 0)) / 2
        potential_gain = ((target / current_price) - 1) * 100 if current_price > 0 else 0
        
        message = (
            f"🔔 DOUBLE BOTTOM BREAKOUT: {symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Pattern Strength: {strength}/100\n"
            f"Current Price:    ${current_price:.2f}\n"
            f"Neckline:         ${neckline:.2f} (BROKEN)\n"
            f"Support Level:    ${support:.2f}\n"
            f"Target Price:     ${target:.2f}\n"
            f"Potential Gain:   {potential_gain:+.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Volume Confirmed: {'✓' if pattern.get('volume_confirmation') else '✗'}\n"
            f"RSI Divergence:   {'✓' if pattern.get('rsi_divergence') else '✗'}"
        )
        
        return message
    
    def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert through all registered handlers.
        
        Args:
            alert: Alert object to send
            
        Returns:
            True if alert was sent successfully
        """
        pattern_key = self._get_pattern_key(alert.pattern_data)
        
        if pattern_key in self.sent_alerts:
            return False
        
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error in alert handler: {e}")
        
        alert.sent = True
        self.sent_alerts.add(pattern_key)
        
        return True
    
    def send_all_pending(self) -> int:
        """
        Send all pending (unsent) alerts.
        
        Returns:
            Number of alerts sent
        """
        sent_count = 0
        
        for alert in self.alerts:
            if not alert.sent:
                if self.send_alert(alert):
                    sent_count += 1
        
        return sent_count
    
    def _console_handler(self, alert: Alert):
        """Default handler that prints alerts to console."""
        priority_emoji = {
            'high': '🔴',
            'normal': '🟡',
            'low': '🟢'
        }
        
        emoji = priority_emoji.get(alert.priority, '⚪')
        
        print("\n" + "="*50)
        print(f"{emoji} ALERT [{alert.priority.upper()}] - {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        print(alert.message)
        print("="*50 + "\n")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get a summary of all alerts."""
        return {
            'total_alerts': len(self.alerts),
            'sent_alerts': sum(1 for a in self.alerts if a.sent),
            'pending_alerts': sum(1 for a in self.alerts if not a.sent),
            'high_priority': sum(1 for a in self.alerts if a.priority == 'high'),
            'symbols_alerted': list(set(a.symbol for a in self.alerts))
        }
    
    def export_alerts(self, filename: str = 'alerts.json'):
        """
        Export all alerts to a JSON file.
        
        Args:
            filename: Output filename
        """
        alerts_data = [alert.to_dict() for alert in self.alerts]
        
        with open(filename, 'w') as f:
            json.dump(alerts_data, f, indent=2, default=str)
        
        print(f"Alerts exported to: {filename}")
    
    def clear_alerts(self):
        """Clear all alerts and reset the manager."""
        self.alerts = []
        self.sent_alerts = set()
        self.pattern_cache = {}


def create_webhook_handler(webhook_url: str) -> Callable[[Alert], None]:
    """
    Create a webhook alert handler.
    
    Args:
        webhook_url: URL to send POST requests to
        
    Returns:
        Handler function
    """
    import requests
    
    def handler(alert: Alert):
        try:
            payload = alert.to_dict()
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Webhook error: {e}")
    
    return handler


def check_and_alert(patterns: List[Dict], 
                   alert_manager: Optional[AlertManager] = None) -> AlertManager:
    """
    Convenience function to check patterns and send alerts.
    
    Args:
        patterns: List of pattern dictionaries
        alert_manager: Optional existing AlertManager
        
    Returns:
        AlertManager instance with alerts
    """
    if alert_manager is None:
        alert_manager = AlertManager()
    
    new_alerts = alert_manager.check_for_alerts(patterns)
    
    for alert in new_alerts:
        alert_manager.send_alert(alert)
    
    return alert_manager
