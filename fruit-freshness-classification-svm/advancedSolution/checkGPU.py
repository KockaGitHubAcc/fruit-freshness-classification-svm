import tensorflow as tf
import os

def check_gpu_health():
    print("🏥 Checking GPU Health...")
    
    # 1. Check if GPU is visible
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ CRITICAL: No GPUs found! Did you restart the runtime?")
        return False

    try:
        # 2. Attempt a simple calculation
        # If the GPU is 'zombie', this specific line will crash immediately
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            
        print(f"✅ GPU is HEALTHY and responsive.")
        print(f"   - Computation result: {c.numpy().tolist()}")
        print(f"   - Device: {gpus[0].name}")
        return True
        
    except RuntimeError as e:
        print("\n💀 GPU IS STILL IN ZOMBIE MODE.")
        print("   Error Details:", e)
        print("\n👉 ACTION REQUIRED: You must restart the entire Jupyter server or container, not just the kernel.")
        return False
    except Exception as e:
        print(f"\n⚠️ Unexpected Error: {e}")
        return False

# Run the check
is_gpu_ok = check_gpu_health()

# Only proceed if True
if not is_gpu_ok:
    raise SystemExit("Stopping execution due to GPU failure.")