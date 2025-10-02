from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
supabase = create_client(supabase_url, supabase_key)

# Upload the model file
with open('best.pt', 'rb') as f:
    model_data = f.read()
    
# Upload to a new bucket called 'models'
try:
    response = supabase.storage.from_('models').upload('best.pt', model_data)
    print("Model uploaded successfully!")
    
    # Get the public URL
    public_url = supabase.storage.from_('models').get_public_url('best.pt')
    print(f"\nModel public URL: {public_url}")
    
    # Update .env file with the URL
    with open('.env', 'a') as env_file:
        env_file.write(f'\nMODEL_URL={public_url}\n')
        print("\nAdded MODEL_URL to .env file")
        
except Exception as e:
    print(f"Error: {str(e)}")
