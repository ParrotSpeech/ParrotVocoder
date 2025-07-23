import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

def export_hifigan_to_onnx(model, output_path, mel_channels=80, max_length=1000):
    """
    Export HiFi-GAN vocoder to ONNX format
    
    Args:
        model: HiFi-GAN generator model (torch.nn.Module)
        output_path: Path to save ONNX model
        mel_channels: Number of mel spectrogram channels (default: 80)
        max_length: Maximum sequence length for dynamic shapes
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input (mel spectrogram)
    # Shape: [batch_size, mel_channels, sequence_length]
    dummy_input = torch.randn(1, mel_channels, 128)
    
    # Define input and output names
    input_names = ["mel_spectrogram"]
    output_names = ["audio_waveform"]
    
    # Define dynamic axes for variable-length sequences
    dynamic_axes = {
        "mel_spectrogram": {2: "sequence_length"},  # Dynamic sequence dimension
        "audio_waveform": {2: "audio_length"}       # Dynamic audio length
    }
    
    # Export options for latest PyTorch
    export_options = {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": 17,  # Latest stable ONNX opset
        "do_constant_folding": True,  # Optimize constant operations
        "export_params": True,  # Include model parameters
        "keep_initializers_as_inputs": False,
        "operator_export_type": torch.onnx.OperatorExportType.ONNX,
        "verbose": False
    }
    
    # For PyTorch 2.1+, use the new dynamo-based exporter (optional)
    try:
        # New dynamo exporter (PyTorch 2.1+)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            dynamo=True,  # Use new exporter
            **export_options
        )
        print(f"Model exported using dynamo exporter to: {output_path}")
    except Exception as e:
        print(f"Dynamo export failed, falling back to TorchScript: {e}")
        # Fallback to TorchScript-based exporter
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            **export_options
        )
        print(f"Model exported using TorchScript exporter to: {output_path}")

def load_and_export_hifigan(checkpoint_path, config=None):
    """
    Load HiFi-GAN model from checkpoint and export to ONNX
    
    Args:
        checkpoint_path: Path to HiFi-GAN checkpoint
        config: Model configuration (if needed)
    """
    
    # Example loading (adjust based on your model structure)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model (replace with your actual model class)
    # model = HiFiGANGenerator(config)  # Your model initialization
    # model.load_state_dict(checkpoint['generator'])
    
    # For demonstration, using a placeholder
    # Replace this with your actual model loading code
    print("Please replace this section with your actual HiFi-GAN model loading code")
    
    # Export to ONNX
    output_path = str(Path(checkpoint_path).with_suffix('.onnx'))
    # export_hifigan_to_onnx(model, output_path)

def verify_onnx_model(onnx_path, test_input_shape=(1, 80, 128)):
    """
    Verify the exported ONNX model
    
    Args:
        onnx_path: Path to ONNX model
        test_input_shape: Shape of test input tensor
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model is valid: {onnx_path}")
        
        # Test with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        
        # Create test input
        test_input = np.random.randn(*test_input_shape).astype(np.float32)
        
        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"✓ ONNX Runtime inference successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {ort_outputs[0].shape}")
        
        return True
        
    except ImportError:
        print("Please install onnx and onnxruntime for verification:")
        print("pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Example: Export a simple model for demonstration
    # Replace this with your actual HiFi-GAN model
    
    class DummyHiFiGAN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(80, 256, 7, padding=3)
            self.conv2 = nn.Conv1d(256, 512, 7, padding=3)
            self.conv3 = nn.Conv1d(512, 1, 7, padding=3)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.tanh(self.conv3(x))
            # Upsample to match HiFi-GAN output size
            x = torch.nn.functional.interpolate(x, scale_factor=256, mode='linear')
            return x
    
    # Create and export dummy model
    dummy_model = DummyHiFiGAN()
    export_hifigan_to_onnx(dummy_model, "hifigan_dummy.onnx")
    
    # Verify the exported model
    verify_onnx_model("hifigan_dummy.onnx")
    
    print("\nTo use with your actual HiFi-GAN model:")
    print("1. Replace DummyHiFiGAN with your actual model class")
    print("2. Load your trained checkpoint")
    print("3. Call export_hifigan_to_onnx() with your model")
    