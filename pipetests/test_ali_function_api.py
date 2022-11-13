import fc2

serverless_client = fc2.Client(
            endpoint = "https://1333298343766298.cn-hangzhou.fc.aliyuncs.com",
            accessKeyID = "LTAI5tRs9g2i4XPHJ15dCZhD",
            accessKeySecret = "hJqoO5yvGJKws7sHxQ3n3w2t17UKAv")

func_fmt = "func_test_{}m"
memory = [4096, 16384, 32768]
code_file = "./func_test_3072m-code.zip"

for mem in memory:
    func_name = func_fmt.format(mem)
    res = serverless_client.create_function('funcpipe', func_name, 'python3',  'test_cloud_launch.handler', timeout=300, memorySize=1024, codeZipFile = code_file)
    print(res)

