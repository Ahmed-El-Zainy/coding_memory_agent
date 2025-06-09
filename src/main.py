
import argparse
import subprocess
import os
import sys
import time
import signal
from multiprocessing import Process

def run_backend(host="0.0.0.0", port=8000):
    cmd = [sys.executable, "src/fastapi_server.py", "--host", host, "--port", str(port)]
    subprocess.run(cmd)

def run_frontend(backend_url="http://localhost:8000", host="0.0.0.0", port=7860, share=False):
    cmd = [sys.executable, "src/gradio_demo.py", "--backend-url", backend_url, "--host", host, "--port", str(port)]
    if share:
        cmd.append("--share")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run chatbot with memory system")
    parser.add_argument("--mode", choices=["backend", "frontend", "both"], default="both")
    parser.add_argument("--backend-host", default="0.0.0.0")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-host", default="0.0.0.0")
    parser.add_argument("--frontend-port", type=int, default=7860)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--share", action="store_true", help="Share Gradio interface publicly")
    
    args = parser.parse_args()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is required")
        sys.exit(1)
    
    processes = []
    
    try:
        if args.mode in ["backend", "both"]:
            print(f"Starting backend on {args.backend_host}:{args.backend_port}")
            backend_process = Process(
                target=run_backend,
                args=(args.backend_host, args.backend_port)
            )
            backend_process.start()
            processes.append(backend_process)
            
            if args.mode == "both":
                time.sleep(3)
        
        if args.mode in ["frontend", "both"]:
            print(f"Starting frontend on {args.frontend_host}:{args.frontend_port}")
            frontend_process = Process(
                target=run_frontend,
                args=(args.backend_url, args.frontend_host, args.frontend_port, args.share)
            )
            frontend_process.start()
            processes.append(frontend_process)
        
        for process in processes:
            process.join()
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()

if __name__ == "__main__":
    main()