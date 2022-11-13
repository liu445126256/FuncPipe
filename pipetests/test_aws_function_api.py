import boto3

lambda_client = boto3.client("lambda")


def config_profiler():
    func_fmt = "func_profiler_{}m"
    memory = [1024, 2048, 3072, 4096, 5120, 6144, 8192, 10240]

    for mem in memory:
        print("Config profiler {}M".format(mem))
        response = lambda_client.update_function_code(
            FunctionName=func_fmt.format(mem),
            S3Bucket='funcpipe',
            S3Key="func_profiler_function.zip",
        )

    '''
    response = lambda_client.update_function_code(
            FunctionName="profiler",
            S3Bucket='funcpipe',
            S3Key="func-pipe.zip",
    )


    for mem in memory:
        print("Config {}M".format(mem))
        response = lambda_client.update_function_configuration(
            FunctionName=func_fmt.format(mem),
            Handler='profiler.handler',
            Timeout=120,
            MemorySize=mem,
            Layers=[
                'arn:aws:lambda:us-east-1:733853814798:layer:psutil:1',
                'arn:aws:lambda:us-east-1:733853814798:layer:requests:1',
                'arn:aws:lambda:us-east-1:733853814798:layer:torch1-5:1',
                'arn:aws:lambda:us-east-1:733853814798:layer:tqdm:1'
            ],
            VpcConfig={
                'SubnetIds': [
                ],
                'SecurityGroupIds': [
                ]
            },
        )

    response = lambda_client.update_function_configuration(
            FunctionName="profiler",
            Handler='profiler_trigger.handler',
            Timeout=600,
            MemorySize=4096,
            Layers=[
                'arn:aws:lambda:us-east-1:733853814798:layer:psutil:1',
                'arn:aws:lambda:us-east-1:733853814798:layer:requests:1',
                'arn:aws:lambda:us-east-1:733853814798:layer:torch1-5:1',
                'arn:aws:lambda:us-east-1:733853814798:layer:tqdm:1'
            ],
            VpcConfig={
                'SubnetIds': [
                ],
                'SecurityGroupIds': [
                ]
            },
        )
    '''
    print("Done")


def create_trainer():
    func_fmt = "func_test_{}m"
    memory = [1024, 2048, 3072, 4096, 5120, 6144, 8192, 10240]
    '''
    for mem in memory:
        print("Creating trainer {}M".format(mem))
        response = lambda_client.create_function(
            FunctionName=func_fmt.format(mem),
            Runtime= 'python3.6',
            Role='arn:aws:iam::733853814798:role/service-role/hybrid_trigger-role-ih44eofo',
            Handler='pipeline.handler',
            Code={
                'S3Bucket': 'funcpipe',
                'S3Key': "func-pipe.zip",
            },
            Timeout=360,
            MemorySize=mem,
            VpcConfig={
                'SubnetIds': [
                ],
                'SecurityGroupIds': [
                ]
            },
            PackageType='Zip',
            Layers=[
                'arn:aws:lambda:us-east-1:733853814798:layer:psutil:1',
                'arn:aws:lambda:us-east-1:733853814798:layer:requests:1',
                'arn:aws:lambda:us-east-1:733853814798:layer:torch1-5:1',
                'arn:aws:lambda:us-east-1:733853814798:layer:tqdm:1'
            ],
        )
    '''
    response = lambda_client.create_function(
        FunctionName="func_test_trigger",
        Runtime='python3.6',
        Role='arn:aws:iam::733853814798:role/service-role/hybrid_trigger-role-ih44eofo',
        Handler='pipeline_trigger.handler',
        Code={
            'S3Bucket': 'funcpipe',
            'S3Key': "func-pipe.zip",
        },
        Timeout=60,
        MemorySize=8192,
        VpcConfig={
            'SubnetIds': [
            ],
            'SecurityGroupIds': [
            ]
        },
        PackageType='Zip',
        Layers=[
            'arn:aws:lambda:us-east-1:733853814798:layer:psutil:1',
            'arn:aws:lambda:us-east-1:733853814798:layer:requests:1',
            'arn:aws:lambda:us-east-1:733853814798:layer:torch1-5:1',
            'arn:aws:lambda:us-east-1:733853814798:layer:tqdm:1'
        ],
    )

def create_hybrid():
    response = lambda_client.create_function(
        FunctionName="Hybrid-pickle",
        Runtime='python3.6',
        Role='arn:aws:iam::733853814798:role/service-role/hybrid_trigger-role-ih44eofo',
        Handler='dl_hybrid.handler',
        Code={
            'S3Bucket': 'funcpipe',
            'S3Key': "Hybrid-pickle-code.zip",
        },
        Timeout=360,
        MemorySize=8192,
        VpcConfig={
            'SubnetIds': [
            ],
            'SecurityGroupIds': [
            ]
        },
        PackageType='Zip',
        Layers=[
            'arn:aws:lambda:us-east-1:733853814798:layer:psutil:1',
            'arn:aws:lambda:us-east-1:733853814798:layer:requests:1',
            'arn:aws:lambda:us-east-1:733853814798:layer:torch1-5:1',
            'arn:aws:lambda:us-east-1:733853814798:layer:tqdm:1'
        ],
    )


def config_worker():
    func_fmt = "func_test_{}m"
    memory = [1024, 2048, 3072, 4096, 5120, 6144, 8192, 10240]

    for mem in memory:
        print("Config worker {}M".format(mem))
        response = lambda_client.update_function_code(
            FunctionName=func_fmt.format(mem),
            S3Bucket='funcpipe',
            S3Key="func_worker_function.zip",
        )

#config_profiler()
config_worker()