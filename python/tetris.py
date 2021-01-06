import random
import datetime
import numpy as np
import copy
# Consistent Var
WIDTH = 10
HEIGHT = 20
# Global Var

class Tertris():
    # Initialization
    def __init__(self,width,height):
        self.reset()
    
    def reset(self):
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
        self.generate_random_blocks(4)
        self.score = 0
        self.alive = True
        return self.get_state()
    # Kill the full line and add empty line on the top then return the eliminated count, return -1 if dead
    def round_finished(self):

        score = 0
        # Check if dead or not

        new_pannel = []
        count = 0
        for i in range(self.height):
            if 0 in self.pannel[i]:
                new_pannel.append(self.pannel[i])
            else:
                count+=1
        for i in range(count):
            line_temp = []
            for j in range(self.width):
                line_temp.append(0)
            new_pannel.insert(0,line_temp)
        self.pannel = new_pannel
        score = count * count
        self.score = self.score + score
        return score

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
        self.current_block = self.simplify_block(pannel_temp)
        # self.current_block = [[1,1],[1,1]] # Test eliminate
        return self.current_block
    
    def simplify_block(self,block):
        def submatrix(a):
            return a[np.min(np.nonzero(a)[0]):np.max(np.nonzero(a)[0])+1,np.min(np.nonzero(a)[1]):np.max(np.nonzero(a)[1])+1]   
        pannel_temp = submatrix(np.array(block) )
        return pannel_temp.tolist()

    def standarlize_block(self,block,force_dim = 0):
         
        pannel_temp = np.array(block) 
        if force_dim !=0:
            pannel_temp =  np.pad(pannel_temp, ((force_dim - pannel_temp.shape[0],0),(force_dim - pannel_temp.shape[1],0)), 'constant')
            
        else:
            max_h_w = max( pannel_temp.shape[0],pannel_temp.shape[1])
            min_h_w = min( pannel_temp.shape[0],pannel_temp.shape[1])
            if pannel_temp.shape[0] - pannel_temp.shape[1] > 0:
                pannel_temp =  np.pad(pannel_temp, ((0,0),(max_h_w - min_h_w,0)), 'constant')
            if pannel_temp.shape[0] - pannel_temp.shape[1] < 0:
                pannel_temp =  np.pad(pannel_temp, ((max_h_w - min_h_w,0),(0,0)), 'constant')
        return pannel_temp.tolist()

    # Rotate
    def rotate(self):
        block_pannel = self.current_block
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
        if self.current_block_x + 1 + len(self.current_block[0]) <= self.width: 
            self.current_block_x = self.current_block_x + 1
            result = True
        return result
    
    # For Block
    def get_lowest_none_zero(self,x):
        block_height = len(self.current_block)
        i = len(self.current_block) -1 
        value = self.current_block[i][x]
        while(value ==0 and i>0):
            i = i-1
            value = self.current_block[i][x]
        return i   

    # For Game Pannel
    def get_highest_none_zero(self,x):
        
        i = 0 
        value = self.pannel[i][x]


        for i in range(self.height):
            value = self.pannel[i][x]
            if value > 0:
                return i

        return self.height
            
        

    # abx aby are the absolute location of x,y of pannel
    def place_block(self,abx,aby,x,y):
        block_height = len(self.current_block)   
        block_width = len(self.current_block[0])
        pannel_temp = copy.deepcopy(self.pannel)
        result = True
        over_top = False
        for i in range(block_height):
            y_temp = block_height - i -1
            y_cor = y_temp - y
            aby_cor = aby + y_cor
            for j in range(block_width):
                x_temp = block_width - j -1
                x_cor = x_temp - x
                abx_cor = abx + x_cor
                # print("searching",aby_cor,abx_cor,y_temp,x_temp)
                if aby_cor >= self.height or abx_cor >= self.width:
                    result = False
                    break
                if aby_cor < 0 or abx_cor < 0:
                    over_top = True
                pannel_temp[aby_cor][abx_cor] = pannel_temp[aby_cor][abx_cor] + self.current_block[y_temp][x_temp]
                if pannel_temp[aby_cor][abx_cor] > 1:
                    result = False
                    break
            
            if result is False:
                break
        if result is False:
            pass
        else:
            self.pannel = pannel_temp
            # Check if it is dead
            if over_top is True:
                # Dead
                self.alive = False
        return result

    def down(self):
        block_width = len(self.current_block[0])
        placed = False
        for x in range(block_width):
            # Find the lowest none zero
            y = self.get_lowest_none_zero(x)
            abs_x = x + self.current_block_x
            abs_y = self.get_highest_none_zero(abs_x) - 1
            # Try to get there
            # print("try",abs_x,abs_y,x,y)
            
            if self.place_block(abs_x,abs_y,x,y) is True:
                placed = True
                break
        if placed is False:
            self.alive = False        
        bonus = self.round_finished()
        print("bonus",bonus)
        # Generat next block
        if bonus >=0 : 
            # Alive
            self.generate_random_blocks(4)
        return bonus

    # For Debug
    def print_pannel(self):
        print("Pannel====")
        for i in range(len(self.pannel)):
            print(self.pannel[i])
    
    def print_block(self):
        print("Block====")
        for i in range(len(self.current_block)):
            print(self.current_block[i])

    # For Reinforcement Learning
    def get_state(self):
        # Pannel State
        pannel_state = []
        for i in range(len(self.pannel)):
            pannel_state = pannel_state + self.pannel[i]
        # Block State
        block_state = []
        block = copy.deepcopy(self.current_block)
        block = self.standarlize_block(block,4)
        #print(block)
        for i in range(len(block)):
            block_state = block_state + block[i]
        return pannel_state + block_state

    # steps = [rotate_times,right_move_steps]
    def step_action(self,steps):
        for i in range(steps[0]):
            self.rotate()
        for i in range(steps[1]):
            self.rightmove()
        bonus = self.down()
        next_state = self.get_state()
        reward = bonus
        done = True
        if self.alive is True:
            done = False
        else:
            reward = 0 - 5 * 5
        info = {
            "pannel":self.pannel,
            "block":self.current_block
        }
        return next_state,reward,done,info
        
    # For presentation
    def render(self):
        pass
    
    def random_action(self):
        action = [random.randint(0, 3),random.randint(0,8)]
        return action

if __name__ == '__main__':
    tertris_test = Tertris(WIDTH,HEIGHT)
    for i in range(1):
        print("NewBlock====")
        block_pannel = tertris_test.generate_random_blocks()
        tertris_test.print_block()

        print("Rotate====")
        block_pannel = tertris_test.rotate()

        print("Rotate====")
        block_pannel = tertris_test.rotate()
        
        print("Rotate====")
        block_pannel = tertris_test.rotate()
        
        print("Rotate====")
        block_pannel = tertris_test.rotate()
        
        #print(tertris_test.get_lowest_none_zero(0)) # Pass
        #print(tertris_test.get_highest_none_zero(0)) # Pass
        # tertris_test.down()

        print("RightMove====")
        # tertris_test.rightmove()
        # tertris_test.rightmove()
        
        #tertris_test.print_block()
        # tertris_test.down()
        # tertris_test.print_pannel()

        #tertris_test.print_block()
        
        # Test Dead
        # while tertris_test.alive is True:
        #     tertris_test.down()
        #     tertris_test.print_pannel()
       

        # Test Eliminate
        '''
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.down()
        tertris_test.print_pannel()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.down()
        tertris_test.print_pannel()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.rightmove()
        tertris_test.down()
        tertris_test.print_pannel()
        '''
        next_state,reward,done,info = tertris_test.step_action([0,4])
        tertris_test.print_pannel()
        #print(len(next_state))
        tertris_test.print_block()