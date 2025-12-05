
import pandas as pd
import numpy as np
from app.utils.plot_generator import PlotGenerator
import sys

# Mocking TSNE to detect calls
from unittest.mock import MagicMock
from sklearn import manifold

# Original TSNE
original_tsne_class = manifold.TSNE

def verify_caching():
    print("🧪 Starting Verification Script...")
    
    # 1. Setup Mock Data
    df = pd.DataFrame({
        'A': np.random.rand(10),
        'B': np.random.rand(10),
        'C': np.random.rand(10),
        'Utility': np.random.rand(10),
        'Row number': range(10)
    })
    input_columns = ['A', 'B', 'C']
    cache_key = "test_key_123"

    # 2. Mock TSNE
    mock_tsne_instance = MagicMock()
    # Return 2D coordinates
    mock_tsne_instance.fit_transform.return_value = np.random.rand(10, 2)
    
    mock_tsne_class = MagicMock(return_value=mock_tsne_instance)
    manifold.TSNE = mock_tsne_class

    try:
        # 3. First Call - Should trigger TSNE
        print("\n🔹 First Call (Uncached)...")
        df1 = PlotGenerator._run_tsne(df.copy(), input_columns, cache_key=cache_key)
        
        if 'tsne-2d-one' in df1.columns:
            print("✅ df1 has TSNE columns")
        else:
            print("❌ df1 missing TSNE columns")
            
        print(f"TSNE call count: {mock_tsne_class.call_count}")
        if mock_tsne_class.call_count != 1:
             print("❌ Expected 1 TSNE call, got", mock_tsne_class.call_count)
             sys.exit(1)
        else:
             print("✅ TSNE called exactly once.")

        # 4. Second Call - Should use Cache (NO TSNE call)
        print("\n🔹 Second Call (Cached)...")
        # Reset mock count? No, we check if it increments.
        current_count = mock_tsne_class.call_count
        
        df2 = PlotGenerator._run_tsne(df.copy(), input_columns, cache_key=cache_key)
        
        if 'tsne-2d-one' in df2.columns:
             print("✅ df2 has TSNE columns")
        
        new_count = mock_tsne_class.call_count
        print(f"TSNE call count after second call: {new_count}")
        
        if new_count == current_count:
            print("✅ TSNE was NOT called again (Cache works!)")
        else:
            print("❌ TSNE was called again (Cache FAILED)")
            sys.exit(1)
            
        # 5. Verify Values Match
        if np.allclose(df1['tsne-2d-one'], df2['tsne-2d-one']):
             print("✅ TSNE values match exactly between calls")
        else:
             print("❌ TSNE values do NOT match")
             sys.exit(1)
             
    finally:
        # Restore TSNE
        manifold.TSNE = original_tsne_class

    print("\n🎉 Verification Successful!")

if __name__ == "__main__":
    verify_caching()
