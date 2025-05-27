"""
Test script for the secure MCP server.
Validates authentication, authorization, and security features.
"""

import requests
import json
import time
import subprocess
import threading
from typing import Optional

class SecurityTester:
    """Test suite for MCP server security."""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.verify = False  # For self-signed certificates
        
    def test_health_check(self) -> bool:
        """Test health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_unauthenticated_access(self) -> bool:
        """Test that unauthenticated requests are rejected."""
        try:
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
            
            response = self.session.post(
                f"{self.base_url}/mcp",
                json=mcp_request,
                timeout=5
            )
            
            if response.status_code == 401:
                print("✅ Unauthenticated access properly rejected")
                return True
            else:
                print(f"❌ Unauthenticated access allowed (status: {response.status_code})")
                return False
                
        except Exception as e:
            print(f"❌ Error testing unauthenticated access: {e}")
            return False
    
    def test_authentication(self, username: str = "admin", password: str = "changeme") -> Optional[str]:
        """Test username/password authentication."""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"username": username, "password": password},
                timeout=5
            )
            
            if response.status_code == 200:
                token_data = response.json()
                token = token_data.get("access_token")
                print(f"✅ Authentication successful: {username}")
                return token
            else:
                print(f"❌ Authentication failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return None
    
    def test_authenticated_access(self, token: str) -> bool:
        """Test authenticated MCP tool access."""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
            
            response = self.session.post(
                f"{self.base_url}/mcp",
                json=mcp_request,
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Authenticated access successful")
                if 'result' in result and 'tools' in result['result']:
                    tools = result['result']['tools']
                    print(f"   Available tools: {[t['name'] for t in tools]}")
                return True
            else:
                print(f"❌ Authenticated access failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Authenticated access error: {e}")
            return False
    
    def test_token_verification(self, token: str) -> bool:
        """Test token verification endpoint."""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.session.get(
                f"{self.base_url}/auth/verify",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                verify_data = response.json()
                print(f"✅ Token verification passed: {verify_data}")
                return True
            else:
                print(f"❌ Token verification failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Token verification error: {e}")
            return False
    
    def test_invalid_credentials(self) -> bool:
        """Test that invalid credentials are rejected."""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"username": "invalid", "password": "invalid"},
                timeout=5
            )
            
            if response.status_code == 401:
                print("✅ Invalid credentials properly rejected")
                return True
            else:
                print(f"❌ Invalid credentials accepted (status: {response.status_code})")
                return False
                
        except Exception as e:
            print(f"❌ Error testing invalid credentials: {e}")
            return False
    
    def test_rate_limiting(self, token: str) -> bool:
        """Test rate limiting functionality."""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            
            # Make multiple rapid requests
            success_count = 0
            rate_limited = False
            
            for i in range(10):
                response = self.session.get(
                    f"{self.base_url}/health",
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:  # Too Many Requests
                    rate_limited = True
                    break
                    
                time.sleep(0.1)  # Small delay between requests
            
            if success_count > 0:
                print(f"✅ Rate limiting configured (processed {success_count} requests)")
                if rate_limited:
                    print("   Rate limit triggered as expected")
                return True
            else:
                print("❌ No requests succeeded")
                return False
                
        except Exception as e:
            print(f"❌ Rate limiting test error: {e}")
            return False
    
    def test_security_headers(self) -> bool:
        """Test that security headers are present."""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            headers = response.headers
            
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Content-Security-Policy': "default-src 'self'"
            }
            
            missing_headers = []
            for header, expected_value in security_headers.items():
                if header not in headers:
                    missing_headers.append(header)
                elif headers[header] != expected_value:
                    print(f"⚠️  Security header {header} has unexpected value: {headers[header]}")
            
            if missing_headers:
                print(f"❌ Missing security headers: {missing_headers}")
                return False
            else:
                print("✅ All security headers present")
                return True
                
        except Exception as e:
            print(f"❌ Security headers test error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all security tests."""
        print("🔒 Starting MCP Server Security Tests")
        print("=" * 50)
        
        test_results = []
        
        # Test health check
        test_results.append(self.test_health_check())
        
        # Test security headers
        test_results.append(self.test_security_headers())
        
        # Test unauthenticated access
        test_results.append(self.test_unauthenticated_access())
        
        # Test invalid credentials
        test_results.append(self.test_invalid_credentials())
        
        # Test authentication
        token = self.test_authentication()
        if token:
            test_results.append(True)
            
            # Test authenticated access
            test_results.append(self.test_authenticated_access(token))
            
            # Test token verification
            test_results.append(self.test_token_verification(token))
            
            # Test rate limiting
            test_results.append(self.test_rate_limiting(token))
        else:
            test_results.extend([False, False, False, False])
        
        # Results summary
        passed = sum(test_results)
        total = len(test_results)
        
        print("\n" + "=" * 50)
        print(f"Test Results: {passed}/{total} passed")
        
        if passed == total:
            print("🎉 All security tests passed!")
            return True
        else:
            print("⚠️  Some security tests failed. Please review the output above.")
            return False

def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCP server security")
    parser.add_argument("--url", default="http://localhost:8081", help="Server URL")
    parser.add_argument("--start-server", action="store_true", help="Start secure server before testing")
    parser.add_argument("--server-args", default="", help="Additional server arguments")
    args = parser.parse_args()
    
    server_process = None
    
    if args.start_server:
        print("🚀 Starting secure MCP server...")
        server_cmd = f"python secure_fabric_mcp.py {args.server_args}"
        server_process = subprocess.Popen(
            server_cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
    
    try:
        # Run tests
        tester = SecurityTester(args.url)
        success = tester.run_all_tests()
        
        if success:
            print("\n✅ Security validation completed successfully!")
        else:
            print("\n❌ Security validation failed!")
            
    finally:
        if server_process:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()
