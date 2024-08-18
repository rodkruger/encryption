from concrete.ml.pandas import ClientEngine
from io import StringIO
import pandas

data_left = """index,total_bill,tip,sex,smoker
1,12.54,2.5,Male,No
2,11.17,1.5,Female,No
3,20.29,2.75,Female,No
"""

# Load your pandas DataFrame
df = pandas.read_csv(StringIO(data_left))

# Obtain client object
client = ClientEngine(keys_path="my_keys")

# Encrypt the DataFrame
df_encrypted = client.encrypt_from_pandas(df)

# Decrypt the DataFrame to produce a pandas DataFrame
df_decrypted = client.decrypt_to_pandas(df_encrypted)