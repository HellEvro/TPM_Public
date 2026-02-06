"""
InfoBot AI Premium License Activation
"""

import sys
import os
from pathlib import Path


def activate_premium_license():
    """Shows Hardware ID and activation instructions"""
    
    print("=" * 60)
    print("InfoBot AI Premium - License Activation")
    print("=" * 60)
    print()
    
    try:
        # In development mode
        if os.getenv('AI_DEV_MODE') == '1':
            print("[DEV MODE] Development mode active - no license needed")
            print()
            print("To use AI modules in dev mode:")
            print("  1. set AI_DEV_MODE=1")
            print("  2. Edit bot_config.py: AI_ENABLED = True")
            print("  3. python bots.py")
            print()
            return
        
        # Try to import license system
        try:
            # Импортируем локальный hardware_id (работает везде)
            import importlib.util
            spec = importlib.util.spec_from_file_location("hardware_id", Path(__file__).parent / "hardware_id.pyc")
            hardware_id = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hardware_id)
            
            get_hardware_id = hardware_id.get_hardware_id
            get_short_hardware_id = hardware_id.get_short_hardware_id
            
            hw_id = get_short_hardware_id()
            full_hw_id = get_hardware_id()
            
            print("=" * 60)
            print("YOUR HARDWARE ID")
            print("=" * 60)
            print()
            print(f"Short HWID: {hw_id}")
            print(f"Full HWID:  {full_hw_id}")
            print()
            print("=" * 60)
            print("ACTIVATION INSTRUCTIONS")
            print("=" * 60)
            print()
            print("1. Send your Hardware ID (use Short HWID above) to:")
            print("   Email: gci.company.ou@gmail.com")
            print()
            print("2. Wait to receive your license file (.lic)")
            print()
            print("3. Place the .lic file in the root folder of InfoBot")
            print("   (Any file with .lic extension will work)")
            print()
            print("4. Restart the bot:")
            print("   python bots.py")
            print()
            print("=" * 60)
            print()
            print("Your license will be automatically detected and activated!")
            print()
            print("IMPORTANT: Send this HWID to gci.company.ou@gmail.com")
            print(f"Short HWID: {hw_id}")
            print()
            
        except ImportError:
            print("[ERROR] Premium license system not found")
            print()
            print("InfoBot AI Premium is not installed.")
            print()
            print("Options:")
            print("  1. Use development mode (free, for testing):")
            print("     set AI_DEV_MODE=1")
            print()
            print("  2. Purchase a license:")
            print("     Email: gci.company.ou@gmail.com")
            print()
            return
    
    except Exception as e:
        print(f"[ERROR] Failed to get hardware ID: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    activate_premium_license()
