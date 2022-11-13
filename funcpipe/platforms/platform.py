'''
Serverless platform APIs, including the basic operations like
    function invoking, storage access ...
'''
from typing import List, Dict

class Platform:
    """ This is a static class, the choice of a specific serverless should be done
    before any operation is called, and it should only be inited once
    """
    platform_type = None
    platform = None

    def __init__(self) -> None:
        raise Exception("This is a static class and should not have any instance.")

    @staticmethod
    def use(platform_name):
        """
        Choose a specific platform
            1. ali cloud
            2. AWS lambda
            3. local environment - for test
        """
        if platform_name == "aws":
            from funcpipe.platforms.aws import AWSPlatform
            Platform.platform = AWSPlatform()
            Platform.platform_type = "aws"

        elif platform_name == "ali":
            from funcpipe.platforms.ali import AliPlatform
            Platform.platform = AliPlatform()
            Platform.platform_type = "ali"

        elif platform_name == "local":
            from funcpipe.platforms.local import LocalPlatform
            Platform.platform = LocalPlatform()
            Platform.platform_type = "local"

        else:
            raise Exception("Specified platform not supported.")

    @staticmethod
    def check_platform_choice():
        if not Platform.platform:
            raise Exception("Platform not specified yet!")

    @staticmethod
    def upload_to_storage(filename: str, data: bytes) -> None:
        Platform.check_platform_choice()
        """Implement the function here"""
        Platform.platform.storage_put(filename, data)

    @staticmethod
    def download_from_storage(filename: str, timeout = -1) -> bytes:
        Platform.check_platform_choice()
        """Implement the function here"""
        return Platform.platform.storage_get(filename, timeout = timeout)

    @staticmethod
    def list_files() -> List:
        Platform.check_platform_choice()
        return Platform.platform.storage_list()

    @staticmethod
    def file_exists(filename: str) -> bool:
        Platform.check_platform_choice()
        return Platform.platform.file_exists(filename)

    @staticmethod
    def delete_from_storage(filename: str) -> None:
        Platform.check_platform_choice()
        Platform.platform.storage_del(filename)

    @staticmethod
    def invoke(launch_info, asynchronous = False):
        Platform.check_platform_choice()
        Platform.platform.invoke(launch_info, asynchronous = asynchronous)

    @staticmethod
    def get_profiler_info() -> Dict:
        Platform.check_platform_choice()
        return Platform.platform.get_profiler_info() # {service_name, function_name}

    @staticmethod
    def set_bucket_name(name = None):
        Platform.check_platform_choice()
        return Platform.platform.set_bucket_name(name)