import subprocess
import time
import datetime
import socket
import sys
import os
from collections import defaultdict
import threading
import ipaddress

class NetworkDeviceMonitor:
    def __init__(self):
        self.known_devices = {}
        self.interface_ips = set()
        self.running = True
        self.refresh_interval = 2  # seconds
        self.first_run = True
        
    def get_local_interfaces(self):
        """Get all local IP addresses"""
        local_ips = set()
        try:
            # Get hostname
            hostname = socket.gethostname()
            # Get all IPs for this host
            for info in socket.getaddrinfo(hostname, None):
                ip = info[4][0]
                if ':' not in ip:  # IPv4 only
                    local_ips.add(ip)
            
            # Also add loopback
            local_ips.add('127.0.0.1')
            
            # Get IPs from ipconfig
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'IPv4 Address' in line or 'Autoconfiguration IPv4 Address' in line:
                    ip = line.split(':')[-1].strip()
                    if ip:
                        local_ips.add(ip)
        except:
            pass
            
        return local_ips
    
    def get_arp_table(self):
        """Get current ARP table entries"""
        devices = {}
        try:
            # Run arp -a command
            result = subprocess.run(['arp', '-a'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            current_interface = None
            
            for line in lines:
                line = line.strip()
                
                # Check if this is an interface line
                if line.startswith('Interface:'):
                    current_interface = line.split()[1]
                    continue
                
                # Skip empty lines and headers
                if not line or 'Internet Address' in line or '---' in line:
                    continue
                
                # Parse ARP entries
                parts = line.split()
                if len(parts) >= 3:
                    ip = parts[0]
                    mac = parts[1]
                    arp_type = parts[2] if len(parts) > 2 else 'unknown'
                    
                    # Skip invalid entries
                    if mac == 'ff-ff-ff-ff-ff-ff' or ip in self.interface_ips:
                        continue
                    
                    # Validate IP
                    try:
                        ipaddress.ip_address(ip)
                        devices[ip] = {
                            'mac': mac,
                            'type': arp_type,
                            'interface': current_interface,
                            'first_seen': datetime.datetime.now(),
                            'last_seen': datetime.datetime.now()
                        }
                    except:
                        continue
                        
        except Exception as e:
            print(f"Error reading ARP table: {e}")
            
        return devices
    
    def get_active_connections(self):
        """Get active network connections"""
        connections = set()
        try:
            # Run netstat command
            result = subprocess.run(['netstat', '-n'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'ESTABLISHED' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        # Extract foreign address
                        foreign = parts[2]
                        if ':' in foreign:
                            ip = foreign.rsplit(':', 1)[0]
                            # Skip IPv6
                            if '[' not in ip and '::' not in ip:
                                try:
                                    ipaddress.ip_address(ip)
                                    connections.add(ip)
                                except:
                                    pass
        except:
            pass
            
        return connections
    
    def resolve_hostname(self, ip):
        """Try to resolve hostname for an IP"""
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            return hostname
        except:
            return None
    
    def classify_device(self, ip, mac=None):
        """Classify device type based on IP and MAC"""
        device_type = "Unknown"
        
        # Check IP ranges
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            if ip_obj.is_private:
                if ip.startswith('192.168.'):
                    device_type = "Local Network Device"
                elif ip.startswith('10.'):
                    device_type = "Corporate Network Device"
                elif ip.startswith('169.254.'):
                    device_type = "Direct Ethernet Connection"
                elif ip.startswith('172.'):
                    octets = ip.split('.')
                    if 16 <= int(octets[1]) <= 31:
                        device_type = "Private Network Device"
            else:
                device_type = "Internet Host"
                
        except:
            pass
            
        # Check MAC address patterns (first 3 octets = manufacturer)
        if mac and mac != 'ff-ff-ff-ff-ff-ff':
            mac_prefix = mac[:8].upper()
            
            # Common manufacturer prefixes
            manufacturers = {
                '00-50-56': 'VMware Virtual Machine',
                '00-0C-29': 'VMware Virtual Machine',
                '00-15-5D': 'Hyper-V Virtual Machine',
                '08-00-27': 'VirtualBox Virtual Machine',
                'AC-DE-48': 'Apple Device',
                '00-1B-63': 'Apple Device',
                'B8-27-EB': 'Raspberry Pi',
                'DC-A6-32': 'Raspberry Pi',
            }
            
            for prefix, mfg in manufacturers.items():
                if mac_prefix.startswith(prefix):
                    device_type = mfg
                    break
                    
        return device_type
    
    def print_device_info(self, ip, info, status):
        """Print formatted device information"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        device_type = self.classify_device(ip, info.get('mac'))
        hostname = self.resolve_hostname(ip)
        
        # Color codes
        colors = {
            'connected': '\033[92m',    # Green
            'disconnected': '\033[91m', # Red
            'active': '\033[93m',       # Yellow
            'info': '\033[94m',         # Blue
            'reset': '\033[0m'          # Reset
        }
        
        status_symbol = '🟢' if status == 'connected' else '🔴' if status == 'disconnected' else '🟡'
        
        print(f"\n{colors[status]}[{timestamp}] {status_symbol} Device {status.upper()}{colors['reset']}")
        print(f"  IP Address:  {ip}")
        if hostname:
            print(f"  Hostname:    {hostname}")
        if info.get('mac'):
            print(f"  MAC Address: {info['mac']}")
        print(f"  Device Type: {device_type}")
        if info.get('interface'):
            print(f"  Interface:   {info['interface']}")
        print("-" * 50)
    
    def monitor(self):
        """Main monitoring loop"""
        print("=" * 60)
        print("  Network Device Monitor - Real-time Connection Tracker")
        print("=" * 60)
        print("\nMonitoring network for device connections...")
        print("Press Ctrl+C to stop\n")
        
        # Get local interfaces
        self.interface_ips = self.get_local_interfaces()
        print(f"Local IP addresses: {', '.join(sorted(self.interface_ips))}\n")
        
        # Initial device scan
        initial_devices = self.get_arp_table()
        active_connections = self.get_active_connections()
        
        # Combine ARP and active connections
        for ip in active_connections:
            if ip not in initial_devices and ip not in self.interface_ips:
                initial_devices[ip] = {
                    'mac': 'unknown',
                    'type': 'active',
                    'first_seen': datetime.datetime.now(),
                    'last_seen': datetime.datetime.now()
                }
        
        if self.first_run:
            print(f"Found {len(initial_devices)} existing device(s):")
            for ip, info in sorted(initial_devices.items()):
                device_type = self.classify_device(ip, info.get('mac'))
                print(f"  • {ip:<15} ({device_type})")
            print("\nNow monitoring for changes...")
            print("-" * 50)
            self.first_run = False
            
        self.known_devices = initial_devices
        
        # Monitor loop
        while self.running:
            try:
                time.sleep(self.refresh_interval)
                
                # Get current devices
                current_devices = self.get_arp_table()
                active_connections = self.get_active_connections()
                
                # Add active connections
                for ip in active_connections:
                    if ip not in current_devices and ip not in self.interface_ips:
                        current_devices[ip] = {
                            'mac': 'unknown',
                            'type': 'active',
                            'first_seen': datetime.datetime.now(),
                            'last_seen': datetime.datetime.now()
                        }
                
                # Check for new devices
                for ip, info in current_devices.items():
                    if ip not in self.known_devices:
                        # New device connected
                        self.known_devices[ip] = info
                        self.print_device_info(ip, info, 'connected')
                        
                        # Play sound alert (Windows)
                        try:
                            import winsound
                            winsound.Beep(1000, 200)  # Frequency, duration
                        except:
                            pass
                
                # Check for disconnected devices
                for ip in list(self.known_devices.keys()):
                    if ip not in current_devices:
                        # Device disconnected
                        info = self.known_devices[ip]
                        self.print_device_info(ip, info, 'disconnected')
                        del self.known_devices[ip]
                        
                        # Play sound alert (Windows)
                        try:
                            import winsound
                            winsound.Beep(500, 200)  # Lower frequency for disconnect
                        except:
                            pass
                            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
    
    def run(self):
        """Start the monitor"""
        try:
            self.monitor()
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
        finally:
            print("\nFinal device summary:")
            print(f"Total devices tracked: {len(self.known_devices)}")
            for ip, info in sorted(self.known_devices.items()):
                device_type = self.classify_device(ip, info.get('mac'))
                print(f"  • {ip:<15} ({device_type})")

def main():
    # Check if running as admin (recommended for better results)
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    
    if not is_admin:
        print("⚠️  Warning: Running without administrator privileges.")
        print("   Some network information may be limited.")
        print("   For best results, run as administrator.\n")
    
    monitor = NetworkDeviceMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
