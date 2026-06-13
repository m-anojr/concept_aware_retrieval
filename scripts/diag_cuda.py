import sys
import traceback

def main():
    print('Python executable:', sys.executable)
    try:
        import torch
        print('torch.__version__:', torch.__version__)
        print('torch.version.cuda:', torch.version.cuda)
        print('torch.cuda.is_available():', torch.cuda.is_available())
        try:
            print('torch.cuda.device_count():', torch.cuda.device_count())
        except Exception as e:
            print('device_count error:', repr(e))
    except Exception as e:
        print('torch import error:')
        traceback.print_exc()

    try:
        import whisper
        print('whisper.__version__:', getattr(whisper, '__version__', 'unknown'))
    except Exception:
        print('whisper import error:')
        traceback.print_exc()

    # Print CUDA_VISIBLE_DEVICES (works on both POSIX and Windows)
    import os
    print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'))

    # Try to run nvidia-smi if available
    try:
        import subprocess
        res = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        print('\n=== nvidia-smi output ===')
        print(res.stdout.strip())
        if res.stderr:
            print('nvidia-smi stderr:', res.stderr.strip())
    except Exception as e:
        print('nvidia-smi run error:', repr(e))

if __name__ == '__main__':
    main()
