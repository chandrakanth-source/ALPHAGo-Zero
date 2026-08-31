import numpy as np
class GoGame:
    def __init__(self,board_size=12):
        self.board_size=board_size
        self.reset()
    def is_legal_move(self, row, col):
        if self.game_over:
            return False
        if not self.is_on_board(row, col):
            return False
        if not self.is_empty(row, col):
            return False
        old_board = self.board.copy()
        self.board[row, col] = self.current_player
        captured = self.capture_opponent_groups(row, col)
        own_group = self.get_group(row, col)
        own_liberties = self.get_liberties(own_group)
        illegal_suicide = (len(own_liberties) == 0 and len(captured) == 0)
        new_hash = self.get_board_hash()
        illegal_ko = new_hash in self.history
        self.board = old_board
        if illegal_suicide:
             return False
        if illegal_ko:
             return False
        return True
    def get_legal_moves(self):
        if self.game_over:
            return []
        legal_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_legal_move(row, col):
                   legal_moves.append((row, col))
        legal_moves.append("PASS")
        return legal_moves
    def reset(self):
        self.board=np.zeros((self.board_size,self.board_size),dtype=np.int8)
        self.current_player=1
        self.consecutive_passes=0
        self.game_over=False
        self.history = set()
        self.history.add(self.get_board_hash())
    def is_terminal(self):
        return self.game_over
    def is_on_board(self,row,col):
        return (0 <= row < self.board_size and 0 <= col < self.board_size)
    def get_neighbors(self,row,col):
        directions=[(-1,0),(1,0),(0,1),(0,-1)]
        neighbors=[]
        for dr,dc in directions:
            r=row+dr
            c=col+dc
            if self.is_on_board(r,c):
                neighbors.append((r,c))
        return neighbors
    def get_board_hash(self):
        return self.board.tobytes()
    def is_empty(self,row,col):
        return self.board[row,col]==0
    def place_stone(self,row,col):
        self.board[row,col]=self.current_player
    def get_group_liberties(self,group):
        return self.get_liberties(group)
    def switch_player(self):
        self.current_player*=-1
    def get_group(self,row,col):
        color=self.board[row,col]
        if color==0:
            return set()
        group=set()
        stack=[(row,col)]
        while stack:
            current=stack.pop()
            if current not in group:
                group.add(current)
                r,c=current
                for nr,nc in self.get_neighbors(r,c):
                    if self.board[nr,nc]==color:
                        if (nr,nc) not in group:
                           stack.append((nr,nc))
        return group
    def get_liberties(self,group):
        liberties=set()
        for r,c in group:
            for nr,nc in self.get_neighbors(r,c):
                if self.board[nr,nc]==0:
                    liberties.add((nr,nc))
        return liberties
    def remove_group(self,group):
        for r,c in group:
            self.board[r,c]=0
    def capture_opponent_groups(self,row,col):
        opponent=-self.current_player
        captured_groups=[]
        for nr,nc in self.get_neighbors(row,col):
            if self.board[nr,nc]==opponent:
                group=self.get_group(nr,nc)
                liberties=self.get_liberties(group)
                if len(liberties)==0:
                    captured_groups.append(group)
        for group in captured_groups:
            self.remove_group(group)
        return captured_groups
    def action_to_position(self, action):
        if action == self.board_size * self.board_size:
            return "PASS"
        row = action // self.board_size
        col = action % self.board_size
        return (row, col)
    def position_to_action(self, row, col):
        if not self.is_on_board(row, col):
          raise ValueError("Position is outside the board")
        return row * self.board_size + col
    def get_pass_action(self):
        return self.board_size * self.board_size
    def get_state(self):

        return {
       "board": self.board.copy(),
        "current_player": self.current_player,
        "consecutive_passes": self.consecutive_passes,
        "game_over": self.game_over,
        "history": self.history.copy()
       }
    def set_state(self, state):
       self.board = state["board"].copy()
       self.current_player = state["current_player"]
       self.consecutive_passes = state["consecutive_passes"]
       self.game_over = state["game_over"]
       self.history = state["history"].copy()
    def make_move(self, row, col=None):
        if col is None:
            if not isinstance(row, int) or row == self.get_pass_action():
                return False
            row, col = divmod(row, self.board_size)
        if self.game_over:
            return False
        if not self.is_on_board(row, col):
            return False
        if not self.is_empty(row, col):
            return False
        old_board=self.board.copy()
        self.board[row,col]=self.current_player
        captured=self.capture_opponent_groups(row, col)
        own_group=self.get_group(row,col)
        own_liberties=self.get_group_liberties(own_group)
        if len(own_liberties)==0 and len(captured)==0:
            self.board=old_board
            return False
        new_hash = self.get_board_hash()
        self.history.add(new_hash)
        self.consecutive_passes=0
        self.switch_player()
        return True
    def is_legal(self, move):
        if move == "PASS":
            return not self.game_over
        if isinstance(move, tuple):
            row, col = move
            return self.is_legal_move(row, col)
        if isinstance(move, int):
            if move == self.get_pass_action():
                return not self.game_over
            row = move // self.board_size
            col = move % self.board_size
            return self.is_legal_move(row, col)
        return False
    def play(self, move):
        if move == "PASS":
            return self.pass_move()
        if isinstance(move, tuple):
            row, col = move
            return self.make_move(row, col)
        if isinstance(move, int):
            if move == self.get_pass_action():
                return self.pass_move()
            row = move // self.board_size
            col = move % self.board_size
            return self.make_move(row, col)
        return False
    def display(self):
        symbols = {0: ".",1: "B",-1: "W"}
        for row in self.board:
           print(" ".join(symbols[int(cell)] for cell in row))
    def count_stones(self):
        black = np.sum(self.board == 1)
        white = np.sum(self.board == -1)
        return int(black), int(white)
    def get_empty_region(self, row, col, visited):
        region = set()
        borders = set()
        stack = [(row, col)]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            r, c = current
            if self.board[r, c] != 0:
               continue
            region.add((r, c))
            for nr, nc in self.get_neighbors(r, c):
               value = self.board[nr, nc]
               if value == 0:
                   if (nr, nc) not in visited:
                      stack.append((nr, nc))
               else:
                      borders.add(value)

        return region, borders
    def calculate_score(self):
        black_score, white_score = self.count_stones()
        visited = set()
        for row in range(self.board_size):
           for col in range(self.board_size):
              if self.board[row, col] != 0:
                continue
              if (row, col) in visited:
                continue
              region, borders = self.get_empty_region(row,col,visited)
              if borders == {1}:
                black_score += len(region)
              elif borders == {-1}:
                white_score += len(region)
        return black_score, white_score
    def get_result(self):
       black_score, white_score = self.calculate_score()
       if black_score > white_score:
          winner = 1
       elif white_score > black_score:
          winner = -1
       else:
          winner = 0
       return {
        "black_score": black_score,
        "white_score": white_score,
        "winner": winner
       }
    def get_winner(self):
       return self.get_result()["winner"]
    def pass_move(self):
        if self.game_over:
           return False
        self.consecutive_passes += 1
        self.switch_player()
        if self.consecutive_passes >= 2:
           self.game_over = True
        return True
