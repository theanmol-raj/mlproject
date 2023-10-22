import sys
def error_message_detail(err,err_detail:sys) -> str:
    _,_,exc_tb =err_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    err_msg = f'Error occured in python script : [{file_name}] line number :[{exc_tb.tb_lineno}] error :[{str(err)}]'
    return err_msg


class CustomException(Exception):
    def __init__(self,error_msg,err_details:sys) -> None:
        super().__init__(error_msg)
        self.error_message = error_message_detail(error_msg,err_details)
    
    def __str__(self) -> str:
        return self.error_message



