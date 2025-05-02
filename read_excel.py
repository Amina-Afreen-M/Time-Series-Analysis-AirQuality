import pandas as pd

try:
    df = pd.read_excel('submission.xlsx')
    
    print("\nFirst 5 rows of submission.xlsx:")
    print(df.head())
    
    print("\nColumns:", list(df.columns))
    print("\nShape:", df.shape)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
except Exception as e:
    print(f"Error reading submission.xlsx: {e}")