#!/usr/bin/env python3
"""
Simple HTTP server to serve the CodeGuard demo interface.
Run this to view the HTML demo in your browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path
import threading
import time


def run_server(port=8080, directory="."):
    """Run HTTP server to serve static files"""
    
    # Change to the specified directory
    os.chdir(directory)
    
    # Create handler
    handler = http.server.SimpleHTTPRequestHandler
    
    # Set up server
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"🚀 CodeGuard Demo Server starting...")
        print(f"📂 Serving files from: {os.getcwd()}")
        print(f"🌐 Server running at: http://localhost:{port}")
        print(f"📄 Demo page: http://localhost:{port}/index.html")
        print(f"📊 Sample report: http://localhost:{port}/reports/")
        print("\n💡 Tips:")
        print("   - Open http://localhost:{port}/index.html to see the demo")
        print("   - View generated reports in the reports/ folder")
        print("   - Press Ctrl+C to stop the server")
        print("\n" + "="*60)
        
        # Try to open browser automatically after a short delay
        def open_browser():
            time.sleep(1)
            try:
                webbrowser.open(f'http://localhost:{port}/index.html')
                print("🌐 Opened demo in browser")
            except:
                print("⚠️  Could not open browser automatically")
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped by user")
            print("👋 Thanks for using CodeGuard!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CodeGuard Demo Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_server.py                 # Start server on port 8080
  python demo_server.py --port 8000     # Start on custom port
  python demo_server.py --no-browser    # Don't open browser automatically
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8080,
        help='Port to run server on (default: 8080)'
    )
    
    parser.add_argument(
        '--directory', '-d',
        type=str,
        default='.',
        help='Directory to serve files from (default: current directory)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Don\'t open browser automatically'
    )
    
    args = parser.parse_args()
    
    # Check if index.html exists
    index_path = Path(args.directory) / 'index.html'
    if not index_path.exists():
        print(f"❌ Error: index.html not found in {args.directory}")
        print("Make sure you're running this from the CodeGuard project directory")
        sys.exit(1)
    
    print("🛡️  CodeGuard - Source Code Plagiarism Detection System")
    print("=" * 60)
    
    try:
        run_server(args.port, args.directory)
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Error: Port {args.port} is already in use")
            print(f"Try using a different port: python demo_server.py --port {args.port + 1}")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()