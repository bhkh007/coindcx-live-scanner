import hmac
import hashlib
import requests
import datetime
import json
import os
import sys
import time
from typing import Optional, Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

class CoinDCXAccountManager:
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        """Initialize CoinDCX Account Manager"""
        
        # Load API credentials
        self.api_key = (api_key or os.getenv("COINDCX_API_KEY", "")).strip()
        self.api_secret = (api_secret or os.getenv("COINDCX_API_SECRET", "")).strip()
        
        # CoinDCX production endpoint
        self.base_url = "https://api.coindcx.com"
        self.is_testnet = False
        
        if not self.api_key or not self.api_secret:
            raise ValueError("COINDCX_API_KEY and COINDCX_API_SECRET must be set in .env file")
    
    def _make_request(self, method: str, endpoint: str, payload: Dict = None):
        url = f"{self.base_url}{endpoint}"

        if payload is None:
            payload = {}

        payload["timestamp"] = int(time.time() * 1000)

        # EXACT payload used for signature
        json_payload = json.dumps(payload, separators=(",", ":"))

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            json_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature
        }

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=json_payload,   # IMPORTANT
                timeout=30
            )

            if response.status_code != 200:
                print("STATUS:", response.status_code)
                print("RESPONSE:", response.text)

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")

            if hasattr(e, "response") and e.response is not None:
                print("Status:", e.response.status_code)
                print("Response:", e.response.text)

            return None
        
    def debug_auth(self):
        payload = {
            "timestamp": int(time.time() * 1000)
        }

        json_payload = json.dumps(payload, separators=(",", ":"))

        signature = hmac.new(
            self.api_secret.encode(),
            json_payload.encode(),
            hashlib.sha256
        ).hexdigest()

        print("API KEY:", self.api_key[:10] + "...")
        print("PAYLOAD:", json_payload)
        print("SIGNATURE:", signature)

        headers = {
            "Content-Type": "application/json",
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature
        }

        response = requests.post(
            "https://api.coindcx.com/exchange/v1/users/info",
            headers=headers,
            data=json_payload
        )

        print(response.status_code)
        print(response.text)
    
    # ============ ACCOUNT INFORMATION ============
    def get_profile(self) -> Optional[Dict]:
        """Get user profile information"""
        return self._make_request("POST", "/exchange/v1/users/info", payload={})
    
    def get_wallet_balances(self) -> Optional[Dict]:
        """Get wallet balances"""
        return self._make_request("POST", "/exchange/v1/users/balances", payload={})
    
    def get_active_orders(self, market: str = None) -> Optional[Dict]:
        """Get active orders"""
        payload = {}
        if market:
            payload["market"] = market
        return self._make_request("POST", "/exchange/v1/orders/active_orders", payload=payload)
    
    def get_trade_history(self, market: str = None) -> Optional[Dict]:
        """Get trade history"""
        payload = {}
        if market:
            payload["market"] = market
        return self._make_request("POST", "/exchange/v1/orders/trade_history", payload=payload)
    
    # ============ BALANCE METHODS ============
    def get_balance_summary(self) -> Dict[str, Any]:
        """Get summarized wallet balances"""

        data = self.get_wallet_balances()

        if data is None:
            return {"error": "Failed to fetch balances"}

        balances = data if isinstance(data, list) else []

        summary = {
            "total_assets": len(balances),
            "non_zero_balances": [],
            "total_blocked": 0.0,
            "total_available": 0.0
        }

        for balance in balances:
            total = float(balance.get("balance", 0))
            locked = float(balance.get("locked_balance", 0))
            available = total - locked
            currency = balance.get("currency")

            summary["total_blocked"] += locked
            summary["total_available"] += available

            if total > 0:
                summary["non_zero_balances"].append({
                    "asset": currency,
                    "balance": total,
                    "available": available,
                    "blocked": locked
                })

        return summary
    
    # ============ UTILITY METHODS ============
    def test_connectivity(self) -> bool:
        """Test API connectivity using public endpoint"""
        try:
            response = requests.get("https://api.coindcx.com/exchange/ticker", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_complete_account_overview(self) -> Dict[str, Any]:
        """Get comprehensive account overview"""
        overview = {
            "timestamp": datetime.datetime.now().isoformat(),
            "profile": self.get_profile(),
            "balances": self.get_balance_summary(),
            "active_orders": self.get_active_orders()
        }
        return overview
    
    def save_account_report(self, filename: str = None) -> str:
        """Save account report to JSON file"""
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"coindcx_report_{timestamp}.json"
        
        overview = self.get_complete_account_overview()
        
        with open(filename, 'w') as f:
            json.dump(overview, f, indent=2, default=str)
        
        print(f"✅ Report saved to: {filename}")
        return filename
    
    # ============ INTERACTIVE MENU ============
    def interactive_menu(self):
        """Interactive CLI menu"""
        print(f"\n💰 CoinDCX Account Manager")
        print(f"🌐 Connected to: {self.base_url}")
        
        if not self.test_connectivity():
            print("⚠️  API connectivity warning, but will attempt authenticated calls")
        
        menu_options = {
            "1": ("👤 Profile Information", self._show_profile),
            "2": ("💰 Wallet Balances", self._show_balances),
            "3": ("📊 Balance Summary", self._show_balance_summary),
            "4": ("📈 Active Orders", self._show_active_orders),
            "5": ("📋 Trade History", self._show_trade_history),
            "6": ("📄 Complete Overview", self._show_complete_overview),
            "7": ("💾 Save Report", self._save_report),
            "0": ("🚪 Exit", None)
        }
        
        while True:
            print("\n" + "="*60)
            print("📋 COINDCX ACCOUNT MANAGEMENT MENU")
            print("="*60)
            for key, (desc, _) in menu_options.items():
                print(f"{key:>2}. {desc}")
            
            choice = input("\n👉 Enter your choice: ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            elif choice in menu_options and menu_options[choice][1]:
                print("\n" + "-"*60)
                menu_options[choice][1]()
            else:
                print("\n❌ Invalid choice")
    
    def _show_profile(self):
        """Display profile"""
        data = self.get_profile()
        if data:
            print("👤 PROFILE INFORMATION")
            print(json.dumps(data, indent=2))
        else:
            print("❌ Failed to fetch profile")
    
    def _show_balances(self):
        """Display balances"""
        data = self.get_wallet_balances()
        if data:
            print("💰 WALLET BALANCES")
            print(json.dumps(data, indent=2))
        else:
            print("❌ Failed to fetch balances")
    
    def _show_balance_summary(self):
        """Display balance summary"""
        summary = self.get_balance_summary()
        if "error" not in summary:
            print("📊 BALANCE SUMMARY")
            for asset in summary.get('non_zero_balances', []):
                print(f"  {asset['asset']}: {asset['balance']:.8f} (Available: {asset['available']:.8f})")
        else:
            print("❌ Failed to fetch balance summary")
    
    def _show_active_orders(self):
        """Display active orders"""
        data = self.get_active_orders()
        if data:
            print("📋 ACTIVE ORDERS")
            print(json.dumps(data, indent=2))
        else:
            print("❌ Failed to fetch orders or none found")
    
    def _show_trade_history(self):
        data = self.get_trade_history()

        if data is None:
            print("❌ Failed to fetch trade history")
            return

        if len(data) == 0:
            print("📈 No trade history found")
            return

        print("📈 TRADE HISTORY")
        print(json.dumps(data, indent=2))
    
    def _show_complete_overview(self):
        """Display complete overview"""
        overview = self.get_complete_account_overview()
        print("📄 COMPLETE ACCOUNT OVERVIEW")
        print(json.dumps(overview, indent=2, default=str))
    
    def _save_report(self):
        """Save report to file"""
        filename = input("Enter filename (press Enter for auto-generated): ").strip()
        self.save_account_report(filename if filename else None)

def main():
    """Main function"""
    try:
        print("KEY FOUND:", bool(os.getenv("COINDCX_API_KEY")))
        print("SECRET FOUND:", bool(os.getenv("COINDCX_API_SECRET")))
        print("🔴 CoinDCX Account Manager - Production Mode")
        manager = CoinDCXAccountManager()
        manager.debug_auth()
        manager.interactive_menu()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Please ensure your .env file contains:")
        print("COINDCX_API_KEY=your_api_key")
        print("COINDCX_API_SECRET=your_api_secret")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()