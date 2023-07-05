### FuncPipe

The following shows how to run FuncPipe on AWS.

---
#### Setup Cloud Enviroment
##### Create Lambda layers
- tqdm
- psutil
- requests
- pytorch==1.5.1
##### Configure bucket name (./funcpipe.conf) 
```
[platform-aws]
bucketName = funcpipe 
```
##### Create S3 bucket
- Install and configure AWS [boto3](https://github.com/boto/boto3)
- Create bucket
```
python3 ./tools/s3_setup.py
```
---
#### Code Upload
##### Dependency
- boto3
- zip
##### Generate Functions
- Configure the Lambda layers and AWS role  in ./tools/aws_function_create.py
```
lambda_layers = [
    '' # arn for each lambda layer
]
role = '' #arn for the role
```
- Run
```
python3 ./tools/aws_function_create.py
```
---
#### Run
```
python3 cli_launch_template.py
```
---
#### Log

##### A. Output to storage 
- Set debugging output in trigger_training.py
```
params["log_type"] = "file"
```
##### B. Output to HTTP server
*Note: this can degrade training performance, for debugging only
- Set debugging output in trigger_training.py
```
params["log_type"] = "http"
```
- Configure HTTP server (VM) IP in funcpipe.conf
```
[logger-http]
url = http://[ip]:5000/logger
```
- Start HTTP server
```
python3 ./tools/flask_server.py
```
---
#### Paper Citation
[FuncPipe: A Pipelined Serverless Framework for Fast and Cost-Efficient Training of Deep Learning Models](https://dl.acm.org/doi/10.1145/3570607)
```
@article{10.1145/3570607,
author = {Liu, Yunzhuo and Jiang, Bo and Guo, Tian and Huang, Zimeng and Ma, Wenhao and Wang, Xinbing and Zhou, Chenghu},
title = {FuncPipe: A Pipelined Serverless Framework for Fast and Cost-Efficient Training of Deep Learning Models},
year = {2022},
volume = {6},
number = {3},
journal = {Proc. ACM Meas. Anal. Comput. Syst.}
}
```
