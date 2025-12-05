import time

class TimeIt:
    """
    A context manager to measure and print nested timing information.
    """
    print_output = True  # Control whether to print timing information
    last_parent = None   # Track the parent TimeIt instance in nested contexts
    level = -1           # Track the current nesting level

    def __init__(self, s):
        """
        Initialize the TimeIt instance.
        :param s: Description of the code block
        """
        self.s = s 
        self.t0 = None  
        self.t1 = None  
        self.outputs = []  
        self.parent = None  

    def __enter__(self):
        """
        Enter the context (start timing).
        """
        self.t0 = time.time()  
        self.parent = TimeIt.last_parent  
        TimeIt.last_parent = self  # Update the last parent to this instance
        TimeIt.level += 1  

    def __exit__(self, t, value, traceback):
        """
        Exit the context (end timing and print information).
        """
        self.t1 = time.time()  
        elapsed_time_ms = (self.t1 - self.t0) * 1000  
        timing_info = '%s%s: %0.1fms' % ('  ' * TimeIt.level, self.s, elapsed_time_ms)  

        TimeIt.level -= 1  # Decrease the nesting level

        if self.parent:
            self.parent.outputs.append(timing_info)
            self.parent.outputs += self.outputs 
        
        else:
            # If this is the top-level block, print the timing information
            if TimeIt.print_output:
                print(timing_info)
                for output in self.outputs:
                    print(output)
            self.outputs = []  # Clear the outputs list

        TimeIt.last_parent = self.parent  # Restore the previous parent

