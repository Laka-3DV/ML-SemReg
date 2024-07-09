import time
from easydict import EasyDict

class Timer:
    def __init__(self) -> None:
        # time record
        self.seg_state = EasyDict()

    def new_start(self, seg_name):
        """
        new and start 
        """
        self.seg_state[seg_name] = EasyDict(start=time.time())
    
    def new_pause(self, seg_name): 
        self.new_start(seg_name)
        self.pause(seg_name)
    
    def pause(self, seg_name):
        self.seg_state[seg_name]["pause"] = time.time()
    
    def resume(self, seg_name):
        if 'pause' in self.seg_state[seg_name]:
            self.seg_state[seg_name].start += (time.time() - self.seg_state[seg_name].pause)
    
    def end(self, seg_name):
        self.resume(seg_name)
        self.seg_state[seg_name].start = time.time() - self.seg_state[seg_name].start
        # self.seg_state[seg_name].start  = self.seg_state[seg_name].start / times
    
    def __repr__(self, times=1) -> str:
        s = ""
        for key, value in self.seg_state.items():
            s += "{}: {:.6f}sec \t".format(key, value.start/times)
        return s
        
    

if __name__=="__main__":
    timer = Timer()

    
    timer.new_start("test1")
    time.sleep(0.34)
    timer.end("test1")

    
    timer.new_start("test2")
    time.sleep(0.34)

    timer.pause("test2")
    time.sleep(0.14)  

    timer.resume("test2")
    time.sleep(0.06)
    timer.end("test2")

    
    timer.new_pause("test3")
    for i in range(10):
        timer.resume("test3")
        time.sleep(0.1)
        timer.pause("test3")
    timer.end("test3")

    print(timer)

"""
python -m eval_taichi.eval_timer
"""