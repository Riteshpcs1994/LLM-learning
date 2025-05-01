import os
import io
import mimetypes
from google.cloud import storage
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

def upload_to_gcs_auto(bucket_name, destination_blob_name, file_path):
    """Automatically detects file type from file_path and uploads it to GCS."""
    
    ext = os.path.splitext(file_path)[1].lower()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    if ext == '.csv':
        df = pd.read_csv(file_path)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        blob.upload_from_string(buffer.getvalue(), content_type='text/csv')

    elif ext == '.parquet':
        df = pd.read_parquet(file_path)
        buffer = io.BytesIO()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='application/octet-stream')

    elif ext == '.log' or ext == '.txt':
        with open(file_path, 'r') as f:
            log_data = f.read()
        blob.upload_from_string(log_data, content_type='text/plain')

    elif ext in ['.png', '.jpg', '.jpeg']:
        img = Image.open(file_path)
        buffer = io.BytesIO()
        img_format = 'PNG' if ext == '.png' else 'JPEG'
        img.save(buffer, format=img_format)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type=f'image/{img_format.lower()}')

    elif ext == '.pt':
        blob.upload_from_filename(file_path, content_type='application/octet-stream')

    else:
        # Fallback to binary upload with guessed MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or 'application/octet-stream'
        blob.upload_from_filename(file_path, content_type=mime_type)

    print(f"Uploaded {file_path} to gs://{bucket_name}/{destination_blob_name}")
