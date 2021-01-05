import random
import datetime
import numpy as np
# Consistent Var
WIDTH = 10
HEIGHT = 20
# Global Var

class Tertris():
    # Initialization
    def __init__(self,width,height):
        self.width = width
        self.height = height
        self.pannel = []
        # Initialize Pannel
        for i in range(height):
            line_temp = []
            for j in range(width):
                line_temp.append(0)
            self.pannel.append(line_temp)
        random.seed(datetime.datetime.now().timestamp())
        self.current_block_x = 0
        self.current_block = self.generate_random_blocks(4)
    # Kill the full line and add empty line on the top then return the eliminated count
    def round_finished(self):
        new_pannel = []
        count = 0
        for i in range(self.height):
            if any(self.pannel[i]) == 0:
                new_pannel.append(self.pannel[i])
            else:
                count+=1
        for i in range(count):
            line_temp = []
            for j in range(self.width):
                line_temp.append(0)
            new_pannel.insert(0,line_temp)
        self.pannel = new_pannel
        return count ** count

    # Generate a volume * volume pannel of block
    def generate_random_blocks(self,volume = 4):
        self.current_block_x = 0
        point = [0,0]
        points = [point]
        pre_dir = random.randint(0,3)
        min_x = 0
        min_y = 0
        dir_list_test = []
        for step in range(volume-1):
            dir = random.randint(0,3)
            # Pick random Point
            point = points[random.randint(0,len(points)-1)]
            invalid = True
            while(invalid is True):
                dir = random.randint(0,3) 
                if dir == (pre_dir+2) % 4:
                    dir = (pre_dir+4) % 4
             
                point_temp = []
                if dir == 0:
                    point_temp = [point[0],point[1]+1]
                if dir == 1:
                    point_temp = [point[0]+1,point[1]]
                if dir == 2:
                    point_temp = [point[0],point[1]-1]
                if dir == 3:
                    point_temp = [point[0]-1,point[1]]
                if point_temp not in points:
                    invalid = False
                    #print(dir)
                    pre_dir = dir
                    points.append(point_temp)
                    point = point_temp
                    dir_list_test.append(dir)
                    break
            
            if point_temp[0] < min_x:
                min_x = point_temp[0] 
            if point_temp[1] < min_y:
                min_y = point_temp[1] 
        points_temp = [] 
        for point_temp in points:
            point_temp[0] = point_temp[0] - min_x
            point_temp[1] = point_temp[1] - min_y
            points_temp.append(point_temp)
        points = points_temp

        pannel_temp = []
        for i in range(volume):
            line_temp = []
            for j in range(volume):
                line_temp.append(0)
            pannel_temp.append(line_temp)
        for point_temp in points:
            pannel_temp[point_temp[0]][point_temp[1]] = 1
        
        return self.simplify_block(pannel_temp)
    
    def simplify_block(self,block):
        def submatrix(a):
            return a[np.min(np.nonzero(a)[0]):np.max(np.nonzero(a)[0])+1,np.min(np.nonzero(a)[1]):np.max(np.nonzero(a)[1])+1]   
        pannel_temp = submatrix(np.array(block) )
        return pannel_temp.tolist()

    def standarlize_block(self,block):
         
        pannel_temp = np.array(block) 
        max_h_w = max( pannel_temp.shape[0],pannel_temp.shape[1])
        min_h_w = min( pannel_temp.shape[0],pannel_temp.shape[1])
        if pannel_temp.shape[0] - pannel_temp.shape[1] > 0:
            pannel_temp =  np.pad(pannel_temp, ((0,0),(max_h_w - min_h_w,0)), 'constant')
        if pannel_temp.shape[0] - pannel_temp.shape[1] < 0:
            pannel_temp =  np.pad(pannel_temp, ((max_h_w - min_h_w,0),(0,0)), 'constant')
        return pannel_temp.tolist()

    # Rotate
    def rotate(self,block_pannel):
        block_pannel = self.standarlize_block(block_pannel)
        total_row = len(block_pannel)
        real_matrix = [block_pannel[col][total_row - 1 - row] for row in range(total_row -1, -1, -1) for col in range(total_row - 1, -1, -1)]
        for row in range(total_row):
            for col in range(total_row):
                block_pannel[row][col] = real_matrix[row*total_row+col]
        self.current_block = self.simplify_block(block_pannel)
        # Check the boarder
        while self.current_block_x  + len(self.current_block[0]) >= self.width: 
            self.current_block_x = self.current_block_x - 1
        return self.current_block

    def rightmove(self):
        result = False
        # Check the boarder
        if self.current_block_x + 1 + len(self.current_block[0]) < self.width: 
            self.current_block_x = self.current_block_x + 1
            result = True
        return result
    
    def down(self):
        

        
        bonus = self.round_finished()
        # Generat next block
        if bonus >=0 : 
            # Alive
            self.generate_random_blocks(4)
        return bonus

if __name__ == '__main__':
    tertris_test = Tertris(WIDTH,HEIGHT)
    for i in range(1):
        block_pannel = tertris_test.generate_random_blocks()
        for i in range(len(block_pannel)):
            print(block_pannel[i])
        print("====")
        block_pannel = tertris_test.rotate(block_pannel)
        for i in range(len(block_pannel)):
            print(block_pannel[i])
