from bot_engine.ai.ai_data_storage import AIDataStorage
import tempfile
import os
import shutil


temp_dir = tempfile.mkdtemp(prefix='ai_diag_')
print('Using temp dir:', temp_dir)
storage = AIDataStorage(data_dir=temp_dir)
storage.add_training_record({'samples': 42, 'duration': 1.23, 'notes': 'diag'})
print('File exists:', os.path.exists(os.path.join(temp_dir, 'ai_training_history.json')))
with open(os.path.join(temp_dir, 'ai_training_history.json'), 'r', encoding='utf-8') as f:
    print('Content:', f.read())
shutil.rmtree(temp_dir, ignore_errors=True)
