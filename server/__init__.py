"""
XenMiner Hashpower Marketplace - Server Package

Mock platform server for offline testing of the hashpower marketplace.
Includes MQTT broker, SQLite storage, REST API, chain simulator, and dashboard.
"""

__version__ = "0.2.0"

__all__ = [
    "account",
    "auth",
    "broker",
    "chain_simulator",
    "dashboard",
    "matcher",
    "pricing",
    "reputation",
    "server",
    "settlement",
    "simulator",
    "storage",
    "watcher",
]
