import logging
import boto3
from botocore.exceptions import ClientError
import os
import zipfile
from funcpipe.configs import Config

lambda_layers = [
    '' # arn for each lambda layer
]
role = '' #arn for the role

lambda_client = boto3.client("lambda")
memory = [1024, 2048, 3072, 4096, 5120, 6144, 8192, 10240] #MB

def create_zip_file(source_dir, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if 'zip' in file: continue
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), source_dir))
    print('[+] zip file created！')

def zip_code(zip_file_name):
    code_path = os.getcwd().strip('tools')
    zip_file_path = code_path + '/{}'.format(zip_file_name)
    if os.path.exists(zip_file_path): os.remove(zip_file_path)
    print("Path: ", code_path)
    create_zip_file(code_path, zip_file_path)
    return zip_file_path

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    print("Uploading zip file...")
    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    print("zip file uploaded.")
    return True

def _create_function(name, handler, bucket_name, zip_file_name, mem_size):
    response = lambda_client.create_function(
        FunctionName= name,
        Runtime='python3.9',
        Role= role,
        Handler= handler,
        Code={
            'S3Bucket': bucket_name,
            'S3Key': zip_file_name,
        },
        Timeout=60,
        MemorySize= mem_size,
        VpcConfig={
            'SubnetIds': [
            ],
            'SecurityGroupIds': [
            ]
        },
        PackageType='Zip',
        Layers= lambda_layers,
    )


def create_worker(function_name, bucket_name, zip_file_name):
    print("Creating worker functions...")
    _create_function(function_name, "train.handler", bucket_name, zip_file_name, 10240)
    function_name_fmt = function_name + "_{}m"
    for mem in memory:
        _create_function(function_name_fmt.format(mem), "train.handler", bucket_name, zip_file_name, mem)
    print("Done.")


def creater_profiler(profiler_name_fmt, bucket_name, zip_file_name):
    print("Creating profiler functions...")
    for mem in memory:
        _create_function(profiler_name_fmt.format(mem), "profiler.handler", bucket_name, zip_file_name, mem)
    print("Done.")

if __name__ == "__main__":
    zip_file_name = 'funcpipe-src.zip'
    bucket_name = Config.getvalue("platform-aws", "bucketName")
    function_name = Config.getvalue("platform-aws", "trainer_function_name")
    profiler_name_fmt = Config.getvalue("platform-aws", "profiler_function_name_fmt")
    # pack the code
    zip_file_path = zip_code(zip_file_name)
    # upload the zip file to s3
    upload_file(zip_file_path, bucket_name)
    # create function
    create_worker(function_name, bucket_name, zip_file_name)
    creater_profiler(profiler_name_fmt, bucket_name, zip_file_name)
    print("All ready to go!")