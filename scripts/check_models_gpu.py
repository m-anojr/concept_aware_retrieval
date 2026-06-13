import traceback

def try_whisper():
    try:
        import whisper
        m = whisper.load_model('small', device='cuda')
        import torch
        print('Whisper loaded on', next(m.parameters()).device)
    except Exception:
        print('Whisper check failed:')
        traceback.print_exc()

def try_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        try:
            import torch
            print('SBERT device:', next(m._first_module().parameters()).device)
        except Exception:
            print('SBERT: could not inspect parameters, instance created')
    except Exception:
        print('SBERT check failed:')
        traceback.print_exc()

def try_clip():
    try:
        from transformers import CLIPModel
        import torch
        m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cuda')
        print('CLIP device:', next(m.parameters()).device)
    except Exception:
        print('CLIP check failed:')
        traceback.print_exc()

def try_easyocr():
    try:
        import easyocr
        try:
            r = easyocr.Reader(['en'], gpu=True)
            print('EasyOCR reader created with gpu=True')
        except Exception:
            print('EasyOCR: reader creation with gpu=True failed:')
            traceback.print_exc()
    except Exception:
        print('EasyOCR import failed:')
        traceback.print_exc()

if __name__ == '__main__':
    print('--- Whisper ---')
    try_whisper()
    print('\n--- SBERT ---')
    try_sbert()
    print('\n--- CLIP ---')
    try_clip()
    print('\n--- EasyOCR ---')
    try_easyocr()
