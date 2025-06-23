import os
import sys
import subprocess
import shutil
from pathlib import Path
import socket
import time
import argparse

# Try to import optional libraries
try:
    from smbclient import smbclient
    SMB_AVAILABLE = True
except ImportError:
    SMB_AVAILABLE = False
    print("Warning: smbclient not installed. Install with: pip install smbclient")

try:
    import ftplib
    FTP_AVAILABLE = True
except ImportError:
    FTP_AVAILABLE = False

try:
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    print("Warning: requests not installed. Install with: pip install requests")

# Configuration
TARGET_IP = ""  # Will be set by user input
DESTINATION_DIR = r"C:\Users\Saem1001\Desktop\dimension_connect"

# Directories to skip
SKIP_DIRS = {
    '$Recycle.Bin', 'System Volume Information', 'Config.Msi', 
    'Windows', 'Program Files', 'Program Files (x86)', 'ProgramData'
}

# Directories to include (None means all)
ONLY_DIRS = None

# File extensions to include (set to None to include all)
INCLUDE_EXTENSIONS = None  # e.g., {'.txt', '.pdf', '.doc', '.jpg', '.png'}

# Auto mode flag
AUTO_MODE = False

def ping_host(host):
    """Ping the host to check connectivity"""
    print(f"Pinging {host}...")
    
    # Windows ping command
    ping_cmd = ["ping", "-n", "4", host]
    
    try:
        result = subprocess.run(ping_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Successfully pinged {host}")
            print(result.stdout)
            return True
        else:
            print(f"✗ Failed to ping {host}")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"Error pinging host: {e}")
        return False

def ensure_destination_dir():
    """Create destination directory if it doesn't exist"""
    Path(DESTINATION_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✓ Destination directory ready: {DESTINATION_DIR}")

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def copy_smb_recursive(share_path, remote_path, local_base_path, level=0):
    """Recursively copy files from SMB share"""
    indent = "  " * level
    full_remote_path = f"{share_path}\\{remote_path}" if remote_path else share_path
    
    try:
        items = smbclient.listdir(full_remote_path)
        
        for item in items:
            if item in ['.', '..']:
                continue
                
            # Skip directories in SKIP_DIRS
            if item in SKIP_DIRS:
                print(f"{indent}⏭️  Skipping directory: {item}/")
                continue
            
            # If ONLY_DIRS is set and we're at root level, skip if not in list
            if ONLY_DIRS and level == 0 and item not in ONLY_DIRS:
                continue
                
            item_remote_path = f"{remote_path}\\{item}" if remote_path else item
            full_item_path = f"{share_path}\\{item_remote_path}"
            local_item_path = os.path.join(local_base_path, item_remote_path.replace('\\', os.sep))
            
            try:
                # Check if it's a directory by trying to list it
                smbclient.listdir(full_item_path)
                # It's a directory
                print(f"{indent}📁 {item}/")
                os.makedirs(local_item_path, exist_ok=True)
                # Recursively copy directory contents
                copy_smb_recursive(share_path, item_remote_path, local_base_path, level + 1)
            except Exception:
                # It's a file
                try:
                    # Skip system files
                    if item.lower() in ['pagefile.sys', 'swapfile.sys', 'hiberfil.sys', 'dumpstack.log.tmp']:
                        print(f"{indent}⏭️  Skipping system file: {item}")
                        continue
                    
                    # Check file extension filter
                    if INCLUDE_EXTENSIONS:
                        _, ext = os.path.splitext(item)
                        if ext.lower() not in INCLUDE_EXTENSIONS:
                            continue
                    
                    # Create parent directory if needed
                    os.makedirs(os.path.dirname(local_item_path), exist_ok=True)
                    
                    # Copy the file
                    with smbclient.open_file(full_item_path, mode='rb') as src:
                        with open(local_item_path, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                    
                    # Get file size for display
                    file_size = os.path.getsize(local_item_path)
                    size_str = format_file_size(file_size)
                    print(f"{indent}✓ {item} ({size_str})")
                except Exception as e:
                    print(f"{indent}✗ {item} - {str(e)[:50]}")
                    
    except Exception as e:
        print(f"{indent}✗ Cannot access {remote_path}: {str(e)[:50]}")

def try_smb_fetch():
    """Try to fetch files using SMB/CIFS protocol"""
    if not SMB_AVAILABLE:
        print("✗ SMB client not available")
        return False
    
    print("\nTrying SMB/CIFS connection...")
    
    # Common share names to try
    share_names = ['C$', 'D$', 'ADMIN$', 'Users', 'Public', 'Shared', 'Files', 'Documents']
    
    for share in share_names:
        try:
            # Try anonymous connection first
            share_path = f"\\\\{TARGET_IP}\\{share}"
            print(f"\nTrying share: {share_path}")
            
            # Test access
            files = smbclient.listdir(share_path)
            
            if files:
                print(f"✓ Connected to {share}")
                
                # Ask user if they want to copy everything (unless in auto mode)
                if AUTO_MODE:
                    response = 'y'
                else:
                    response = input(f"\nFound {len(files)} items in {share}. Copy all files recursively? (y/n): ")
                
                if response.lower() == 'y':
                    print(f"\nCopying files from {share}...")
                    local_share_path = os.path.join(DESTINATION_DIR, share.replace('$', '_drive'))
                    os.makedirs(local_share_path, exist_ok=True)
                    
                    copy_smb_recursive(share_path, "", local_share_path)
                    print(f"\n✓ Finished copying from {share}")
                    return True
                else:
                    print("Skipping this share...")
                
        except Exception as e:
            print(f"  Failed to access {share}: {str(e)[:50]}")
    
    return False

def try_ftp_fetch():
    """Try to fetch files using FTP protocol"""
    if not FTP_AVAILABLE:
        print("✗ FTP client not available")
        return False
    
    print("\nTrying FTP connection...")
    
    try:
        # Try anonymous FTP
        ftp = ftplib.FTP()
        ftp.connect(TARGET_IP, 21, timeout=10)
        ftp.login()  # Anonymous login
        
        print("✓ Connected to FTP server")
        
        # List and download files
        files = ftp.nlst()
        
        for file in files:
            if file not in ['.', '..']:
                local_path = os.path.join(DESTINATION_DIR, file)
                print(f"Downloading: {file}")
                
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f'RETR {file}', f.write)
                
                print(f"✓ Downloaded: {file}")
        
        ftp.quit()
        return True
        
    except Exception as e:
        print(f"✗ FTP connection failed: {e}")
        return False

def try_http_fetch():
    """Try to fetch files using HTTP protocol"""
    if not HTTP_AVAILABLE:
        print("✗ Requests library not available")
        return False
    
    print("\nTrying HTTP connection...")
    
    # Try common ports
    ports = [80, 8080, 8000, 3000, 5000]
    
    for port in ports:
        url = f"http://{TARGET_IP}:{port}"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ HTTP server found on port {port}")
                
                # Save the index page
                index_path = os.path.join(DESTINATION_DIR, f"index_port_{port}.html")
                with open(index_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                print(f"✓ Saved index page from port {port}")
                
                # Note: Without directory listing, we can't automatically fetch all files
                print("Note: HTTP directory listing not available. Manual browsing required.")
                return True
                
        except Exception as e:
            continue
    
    print("✗ No HTTP server found")
    return False

def check_open_ports():
    """Scan for open ports on the target"""
    print(f"\nScanning common ports on {TARGET_IP}...")
    
    common_ports = {
        21: "FTP",
        22: "SSH/SFTP",
        23: "Telnet",
        80: "HTTP",
        139: "NetBIOS",
        445: "SMB/CIFS",
        3389: "RDP",
        5900: "VNC",
        8080: "HTTP-Alt"
    }
    
    open_ports = []
    
    for port, service in common_ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        
        result = sock.connect_ex((TARGET_IP, port))
        if result == 0:
            print(f"✓ Port {port} ({service}) is open")
            open_ports.append((port, service))
        
        sock.close()
    
    return open_ports

def get_local_ip():
    """Get the local IP address"""
    try:
        # Create a socket and connect to an external host
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return None

def quick_network_scan(network_prefix, timeout=0.5):
    """Quickly scan for active IPs in the local network"""
    print(f"\nScanning {network_prefix}.0/24 for active hosts...")
    print("(This may take a moment...)")
    
    active_ips = []
    
    # Scan common IPs first
    common_ips = [1, 2, 254, 100, 101, 102, 103, 104, 105]
    
    for last_octet in common_ips:
        ip = f"{network_prefix}.{last_octet}"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        # Try common ports
        for port in [445, 139, 80, 22]:
            result = sock.connect_ex((ip, port))
            if result == 0:
                active_ips.append(ip)
                print(f"  ✓ Found: {ip}")
                break
        
        sock.close()
    
    return active_ips

def suggest_ip_addresses(skip_scan=False):
    """Suggest common IP addresses based on local network"""
    local_ip = get_local_ip()
    suggestions = []
    
    if local_ip:
        # Get network prefix (e.g., 192.168.1.x -> 192.168.1)
        parts = local_ip.split('.')
        network_prefix = '.'.join(parts[:3])
        
        print(f"\nYour local IP: {local_ip}")
        print(f"Network: {network_prefix}.0/24")
        
        # Ask if user wants to scan (unless skip_scan is True)
        if not skip_scan:
            scan_choice = input("\nScan network for active devices? (y/N): ").strip().lower()
            
            if scan_choice == 'y':
                active_ips = quick_network_scan(network_prefix)
                if active_ips:
                    suggestions = active_ips[:5]  # Limit to 5 suggestions
                else:
                    print("No active devices found.")
        
        # Add default suggestions if no scan or no results
        if not suggestions:
            suggestions = [
                f"{network_prefix}.1",    # Common router IP
                f"{network_prefix}.254",  # Alternative router IP
                "169.254.78.159",        # Your default IP
            ]
        
        print(f"\nAvailable IPs:")
        for i, ip in enumerate(suggestions, 1):
            print(f"  {i}. {ip}")
    
    return suggestions

def main():
    global AUTO_MODE, INCLUDE_EXTENSIONS, SKIP_DIRS, ONLY_DIRS, TARGET_IP, DESTINATION_DIR
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fetch files from a network device')
    parser.add_argument('-a', '--auto', action='store_true', help='Run in automatic mode (no prompts)')
    parser.add_argument('-t', '--target', help='Target IP address')
    parser.add_argument('-d', '--destination', help='Destination directory', default=DESTINATION_DIR)
    parser.add_argument('-e', '--extensions', nargs='+', help='File extensions to include (e.g., .txt .pdf)')
    parser.add_argument('--skip-dirs', nargs='+', help='Additional directories to skip')
    parser.add_argument('--only-dirs', nargs='+', help='Only copy these directories')
    parser.add_argument('--no-scan', action='store_true', help='Skip network scan when prompting for IP')
    
    args = parser.parse_args()
    
    # Update configuration
    AUTO_MODE = args.auto
    if args.destination:
        DESTINATION_DIR = args.destination
    
    # Get target IP - either from command line or user input
    if args.target:
        TARGET_IP = args.target
    else:
        print("=== Dimension Connect File Fetcher ===")
        
        # Show IP suggestions
        suggestions = suggest_ip_addresses()
        
        print("\nEnter the target IP address to scan")
        if suggestions:
            print(f"(Enter a number 1-{len(suggestions)} for suggestions above, or type full IP)")
        print("(Press Enter for default: 169.254.78.159)")
        
        user_input = input("\nTarget IP: ").strip()
        
        if not user_input:
            TARGET_IP = "169.254.78.159"
        elif user_input.isdigit() and suggestions and 1 <= int(user_input) <= len(suggestions):
            TARGET_IP = suggestions[int(user_input) - 1]
            print(f"Selected: {TARGET_IP}")
        else:
            TARGET_IP = user_input
        
        # Also ask for destination if not in auto mode
        if not AUTO_MODE:
            print(f"\nEnter destination directory")
            print(f"(Press Enter for default: {DESTINATION_DIR})")
            custom_dest = input("Destination: ").strip()
            if custom_dest:
                DESTINATION_DIR = custom_dest
        print()
    
    # Validate IP address format
    try:
        parts = TARGET_IP.split('.')
        if len(parts) != 4:
            raise ValueError("Invalid IP format")
        for part in parts:
            num = int(part)
            if num < 0 or num > 255:
                raise ValueError("Invalid IP octet")
    except:
        print(f"✗ Invalid IP address format: {TARGET_IP}")
        print("Please use format: xxx.xxx.xxx.xxx")
        sys.exit(1)
    
    if args.extensions:
        INCLUDE_EXTENSIONS = set(ext if ext.startswith('.') else f'.{ext}' for ext in args.extensions)
        print(f"Filtering for extensions: {', '.join(INCLUDE_EXTENSIONS)}")
    
    if args.skip_dirs:
        SKIP_DIRS.update(args.skip_dirs)
    
    if args.only_dirs:
        ONLY_DIRS = set(args.only_dirs)
        print(f"Only copying directories: {', '.join(ONLY_DIRS)}")
    
    # Update destination to include IP address if using default path
    base_default_path = r"C:\Users\Saem1001\Desktop\dimension_connect"
    if DESTINATION_DIR == base_default_path:
        # Default path - add IP to it
        DESTINATION_DIR = os.path.join(base_default_path, TARGET_IP.replace('.', '_'))
    
    print(f"=== Dimension Connect File Fetcher ===")
    print(f"Target IP: {TARGET_IP}")
    print(f"Destination: {DESTINATION_DIR}")
    if AUTO_MODE:
        print("Mode: Automatic (no prompts)")
    print()
    
    # Step 1: Ping the host
    if not ping_host(TARGET_IP):
        print("\n⚠ Warning: Host may be unreachable, but continuing anyway...")
    
    # Step 2: Ensure destination directory exists
    ensure_destination_dir()
    
    # Step 3: Check open ports
    open_ports = check_open_ports()
    
    if not open_ports:
        print("\n✗ No open ports found. The target may be offline or firewalled.")
        return
    
    # Step 4: Try different protocols based on open ports
    success = False
    
    # Try SMB if port 445 or 139 is open
    if any(p[0] in [445, 139] for p in open_ports):
        if try_smb_fetch():
            success = True
    
    # Try FTP if port 21 is open
    if not success and any(p[0] == 21 for p in open_ports):
        if try_ftp_fetch():
            success = True
    
    # Try HTTP if port 80 or 8080 is open
    if not success and any(p[0] in [80, 8080] for p in open_ports):
        if try_http_fetch():
            success = True
    
    if success:
        print(f"\n✓ Files have been saved to: {DESTINATION_DIR}")
        
        # Show summary
        total_files = sum(len(files) for _, _, files in os.walk(DESTINATION_DIR))
        total_size = sum(os.path.getsize(os.path.join(dirpath, f))
                        for dirpath, _, filenames in os.walk(DESTINATION_DIR)
                        for f in filenames)
        
        print(f"\nSummary:")
        print(f"  Total files copied: {total_files}")
        print(f"  Total size: {format_file_size(total_size)}")
    else:
        print("\n✗ Could not fetch files. Manual intervention may be required.")
        print("\nSuggestions:")
        print("1. Check if you need credentials for the target system")
        print("2. Verify the target is sharing files")
        print("3. Check Windows Firewall settings")
        print("4. Try mapping the network drive manually:")
        print(f"   net use Z: \\\\{TARGET_IP}\\ShareName")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)