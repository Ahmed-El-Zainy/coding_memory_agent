
# import argparse
# import subprocess
# import os
# import sys
# import time
# import signal
# from multiprocessing import Process

# def run_backend(host="0.0.0.0", port=8000):
#     cmd = [sys.executable, "src/fastapi_server.py", "--host", host, "--port", str(port)]
#     subprocess.run(cmd)

# def run_frontend(backend_url="http://localhost:8000", host="0.0.0.0", port=7860, share=False):
#     cmd = [sys.executable, "src/gradio_demo.py", "--backend-url", backend_url, "--host", host, "--port", str(port)]
#     if share:
#         cmd.append("--share")
#     subprocess.run(cmd)

# def main():
#     parser = argparse.ArgumentParser(description="Run chatbot with memory system")
#     parser.add_argument("--mode", choices=["backend", "frontend", "both"], default="both")
#     parser.add_argument("--backend-host", default="0.0.0.0")
#     parser.add_argument("--backend-port", type=int, default=8000)
#     parser.add_argument("--frontend-host", default="0.0.0.0")
#     parser.add_argument("--frontend-port", type=int, default=7860)
#     parser.add_argument("--backend-url", default="http://localhost:8000")
#     parser.add_argument("--share", action="store_true", help="Share Gradio interface publicly")
    
#     args = parser.parse_args()
    
#     if not os.getenv("GOOGLE_API_KEY"):
#         print("Error: GOOGLE_API_KEY environment variable is required")
#         sys.exit(1)
    
#     processes = []
    
#     try:
#         if args.mode in ["backend", "both"]:
#             print(f"Starting backend on {args.backend_host}:{args.backend_port}")
#             backend_process = Process(
#                 target=run_backend,
#                 args=(args.backend_host, args.backend_port)
#             )
#             backend_process.start()
#             processes.append(backend_process)
            
#             if args.mode == "both":
#                 time.sleep(3)
        
#         if args.mode in ["frontend", "both"]:
#             print(f"Starting frontend on {args.frontend_host}:{args.frontend_port}")
#             frontend_process = Process(
#                 target=run_frontend,
#                 args=(args.backend_url, args.frontend_host, args.frontend_port, args.share)
#             )
#             frontend_process.start()
#             processes.append(frontend_process)
        
#         for process in processes:
#             process.join()
            
#     except KeyboardInterrupt:
#         print("\nShutting down...")
#         for process in processes:
#             process.terminate()
#         for process in processes:
#             process.join()

# if __name__ == "__main__":
#     main()



import argparse
import subprocess
import os
import sys
import time
import signal
import yaml
from multiprocessing import Process
from pathlib import Path
from logs.custom_logger import CustomLoggerTracker

logging = CustomLoggerTracker()
logger = logging.get_logger(__name__)

def load_config():
    config_path = Path("src/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    return {}

def check_environment():
    config = load_config()
    
    required_keys = ["google_api_key"]
    missing_keys = []
    
    for key in required_keys:
        if not config.get(key) and not os.getenv(key.upper()):
            missing_keys.append(key.upper())
    
    if missing_keys:
        print(f"Error: The following environment variables are required: {', '.join(missing_keys)}")
        print("Please set them in your environment or config.yaml file")
        return False
    
    return True

def install_dependencies():
    try:
        print("Checking and installing dependencies...")
        requirements = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "gradio>=4.0.0",
            "requests>=2.31.0",
            "pydantic>=2.0.0",
            "google-generativeai>=0.3.0",
            "scikit-learn>=1.3.0",
            "numpy>=1.24.0",
            "pillow>=10.0.0",
            "aiofiles>=23.0.0",
            "python-multipart>=0.0.6",
            "pyyaml>=6.0.1"
        ]
        
        for req in requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", req], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                print(f"Warning: Could not install {req}")
        
        print("Dependencies installation completed")
        return True
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_directories():
    directories = [
        "assets/uploads",
        "assets/logs", 
        "assets/chroma_db",
        "assets/chroma_backup"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("Required directories created")

def run_backend(host="0.0.0.0", port=8000, reload=False):
    cmd = [
        sys.executable, "src/fastapi_server.py", 
        "--host", host, 
        "--port", str(port)
    ]
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, cwd=".")
    except KeyboardInterrupt:
        print("\nBackend shutdown requested")
    except Exception as e:
        print(f"Backend error: {e}")

def run_frontend(backend_url="http://localhost:8000", host="0.0.0.0", port=7860, share=False):
    cmd = [
        sys.executable, "src/gradio_demo.py", 
        "--backend-url", backend_url, 
        "--host", host, 
        "--port", str(port)
    ]
    if share:
        cmd.append("--share")
    
    try:
        subprocess.run(cmd, cwd=".")
    except KeyboardInterrupt:
        print("\nFrontend shutdown requested")
    except Exception as e:
        print(f"Frontend error: {e}")

def wait_for_backend(host, port, timeout=30):
    import requests
    import time
    
    url = f"http://{host}:{port}/health"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"Backend is ready at {url}")
                return True
        except:
            pass
        time.sleep(1)
    
    print(f"Backend failed to start within {timeout} seconds")
    return False

def main():
    parser = argparse.ArgumentParser(description="Advanced Chatbot with Memory System")
    parser.add_argument("--mode", choices=["backend", "frontend", "both"], default="both",
                       help="Run backend, frontend, or both services")
    parser.add_argument("--backend-host", default=None,
                       help="Backend host (default from config)")
    parser.add_argument("--backend-port", type=int, default=None,
                       help="Backend port (default from config)")
    parser.add_argument("--frontend-host", default=None,
                       help="Frontend host (default from config)")
    parser.add_argument("--frontend-port", type=int, default=None,
                       help="Frontend port (default from config)")
    parser.add_argument("--backend-url", default=None,
                       help="Backend URL for frontend (auto-generated if not provided)")
    parser.add_argument("--share", action="store_true",
                       help="Share Gradio interface publicly")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install required dependencies")
    parser.add_argument("--no-env-check", action="store_true",
                       help="Skip environment variable checks")
    
    args = parser.parse_args()
    
    if args.install_deps:
        if not install_dependencies():
            sys.exit(1)
    
    if not args.no_env_check and not check_environment():
        sys.exit(1)
    
    setup_directories()
    
    config = load_config()
    
    backend_host = args.backend_host or config.get('api', {}).get('host', '0.0.0.0')
    backend_port = args.backend_port or config.get('api', {}).get('port', 8000)
    frontend_host = args.frontend_host or config.get('ui', {}).get('frontend', {}).get('host', '0.0.0.0')
    frontend_port = args.frontend_port or config.get('ui', {}).get('frontend', {}).get('port', 7860)
    backend_url = args.backend_url or f"http://localhost:{backend_port}"
    
    processes = []
    
    print("=" * 60)
    print("🤖 Advanced Chatbot with Memory System")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    if args.mode in ["backend", "both"]:
        print(f"Backend: http://{backend_host}:{backend_port}")
    if args.mode in ["frontend", "both"]:
        print(f"Frontend: http://{frontend_host}:{frontend_port}")
    print("=" * 60)
    
    try:
        if args.mode in ["backend", "both"]:
            print(f"🚀 Starting backend server...")
            backend_process = Process(
                target=run_backend,
                args=(backend_host, backend_port, args.reload)
            )
            backend_process.start()
            processes.append(("Backend", backend_process))
            
            if args.mode == "both":
                print("⏳ Waiting for backend to be ready...")
                if not wait_for_backend(backend_host, backend_port):
                    print("❌ Backend failed to start. Exiting...")
                    for name, process in processes:
                        process.terminate()
                    sys.exit(1)
        
        if args.mode in ["frontend", "both"]:
            print(f"🎨 Starting frontend interface...")
            frontend_process = Process(
                target=run_frontend,
                args=(backend_url, frontend_host, frontend_port, args.share)
            )
            frontend_process.start()
            processes.append(("Frontend", frontend_process))
            
            if args.share:
                print("🌐 Gradio interface will be shared publicly")
        
        print("\n✅ All services started successfully!")
        print("\n📋 Service URLs:")
        if args.mode in ["backend", "both"]:
            print(f"   • API Documentation: http://{backend_host}:{backend_port}/docs")
            print(f"   • Health Check: http://{backend_host}:{backend_port}/health")
        if args.mode in ["frontend", "both"]:
            print(f"   • Chat Interface: http://{frontend_host}:{frontend_port}")
        
        print("\n💡 Press Ctrl+C to stop all services")
        
        for name, process in processes:
            process.join()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested...")
        for name, process in processes:
            print(f"   Stopping {name}...")
            process.terminate()
        
        print("⏳ Waiting for processes to terminate...")
        for name, process in processes:
            process.join(timeout=5)
            if process.is_alive():
                print(f"   Force killing {name}...")
                process.kill()
        
        print("✅ All services stopped successfully")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        for name, process in processes:
            process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()