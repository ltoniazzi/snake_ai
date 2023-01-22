from pygame.locals import *
from random import randint
import pygame
import time
import pandas as pd


class LogGameState:
    def __init__(self) -> None:
        self.columns = ["PlayerX", "PlayerY", "AppleX", "AppleY", "Direction", "DirectionNew", "Score", "ScoreDiff"]  # Direction after action, add direction pre action
        self.df = pd.DataFrame(columns=self.columns)

    def add_state(self, game_state=None, score=None, current_direction=None):
        # use pandas to save state + action + outcome
        new_state = {
            "PlayerX": [game_state["Player"].x[:3]],
            "PlayerY": [game_state["Player"].y[:3]],
            "AppleX": [game_state["Apple"].y],
            "AppleY": [game_state["Apple"].y],
            "Direction": [current_direction],
            "DirectionNew": [game_state["Player"].direction],
            "Score": [score],
        }
        new_state = pd.DataFrame(new_state, index=[self.df.shape[0]])
        self.df = pd.concat([self.df, new_state], axis=0)

        print(self.df)
    
    def save_states():
        pass


class AgentRuleBased:
    def __init__(self) -> None:

        self.action = None

    def act(self, game_state=None):

        self.action = {
            "K_RIGHT": False,
            "K_LEFT": False,
            "K_UP": False,
            "K_DOWN": False,
        }

        direction = game_state["Player"].direction
        player_x = game_state["Player"].x[0]
        player_y = game_state["Player"].y[0]
        apple_x = game_state["Apple"].x
        apple_y = game_state["Apple"].y

        # print(f"{player_x=}")
        # print(f"{apple_x=}")
        # print(f"{player_y=}")
        # print(f"{apple_y=}")
        # print(f"{direction=}")
        # print(f"\n")

        if player_y <= apple_y:
            print("yforw")
            if player_x >= apple_x -50:
                print("yforw")
                if direction == 0 or direction == 1:
                    self.action["K_DOWN"] = True
                # elif direction == 2 or direction == 3:
                #     self.action["K_LEFT"] = True


class Apple:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, surface, image, dest_rect):
        print("apple", self.x, self.y)
        dest_rect.x = self.x
        dest_rect.y = self.y
        # dest_rect.inflate(-100, -100)
        surface.blit(image, dest_rect)


class Player:
    def __init__(self, length):
        self.step = 44
        self.x = [self.step]
        self.y = [self.step]
        self.direction = 0
        self.length = 3

        self.updateCountMax = 2
        self.updateCount = 0
        self.length = length
        for i in range(0, 200):
            self.x.append(-100)
            self.y.append(-100)

        # initial positions, no collision.
        self.x[1] = self.step - self.step
        self.x[2] = self.step - 2 * self.step
        self.y[1] = self.step
        self.y[2] = self.step

    def update(self):

        self.updateCount = self.updateCount + 1
        if self.updateCount > self.updateCountMax:

            # update previous positions
            for i in range(self.length - 1, 0, -1):
                self.x[i] = self.x[i - 1]
                self.y[i] = self.y[i - 1]

                # print(f"{self.y[:4]=}")
                # print(f"{self.x[:4]=}")

            # update position of head of snake
            if self.direction == 0:
                self.x[0] = self.x[0] + self.step
            if self.direction == 1:
                self.x[0] = self.x[0] - self.step
            if self.direction == 2:
                self.y[0] = self.y[0] - self.step
            if self.direction == 3:
                self.y[0] = self.y[0] + self.step

            self.updateCount = 0

    def moveRight(self):
        self.direction = 0

    def moveLeft(self):
        self.direction = 1

    def moveUp(self):
        self.direction = 2

    def moveDown(self):
        self.direction = 3

    def draw(self, surface, image, dest_rect):
        for i in range(0, self.length):
            if i < 1:
                print(i, ": ", self.x[i], self.y[i])
            dest_rect.x = self.x[i]
            dest_rect.y = self.y[i]
            surface.blit(image, dest_rect)


class Game:
    def isCollision(self, apple_x, apple_y, player_x, player_y, bsize):

        if apple_x >= player_x and apple_x <= player_x + bsize:
            if apple_y >= player_y and apple_y <= player_y + bsize:
                return True

        return False

    def isCollisionWall(self, player_x, player_y, windowWidth, windowHeight):

        if player_x < 0 or player_x > windowWidth:
            return True
        if player_y < 0 or player_y > windowHeight:
            return True

        return False


class App:

    windowWidth = 800
    windowHeight = 600
    player = None
    apple = None

    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.game = Game()
        self.player = Player(3)
        self.apple = Apple(300, 300)
        self.score = 0
        self.scale_img = 1
        self.log_game_state = LogGameState()

        self.agent = AgentRuleBased()
        self.game_state = {
            "Player": self.player,
            "Apple": self.apple,
            "Width": self.windowWidth,
            "Height": self.windowHeight,
            "Score": self.score,
        }

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(
            (self.windowWidth, self.windowHeight), pygame.HWSURFACE
        )

        pygame.display.set_caption(f"Score: {self.score}")
        self._running = True
        self._image_surf = pygame.image.load("snake.jpg").convert()
        self._image_dest_rect = pygame.Rect(
            0,
            0,
            self._image_surf.get_width() // self.scale_img,
            self._image_surf.get_height() // self.scale_img,
        )
        self._apple_surf = pygame.image.load("apple.jpg").convert()
        self._apple_dest_rect = pygame.Rect(
            0,
            0,
            self._apple_surf.get_width() // self.scale_img,
            self._apple_surf.get_height() // self.scale_img,
        )

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self):
        self.player.update()

        if self.game.isCollisionWall(
            self.player.x[0], self.player.y[0], self.windowWidth, self.windowHeight
        ):
            print("You lose! Collision: you hit the wall man")
            print("x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")")
            exit(0)

        # does snake eat apple?
        for i in range(0, self.player.length):
            if self.game.isCollision(
                self.apple.x, self.apple.y, self.player.x[i], self.player.y[i], 44
            ):
                self.apple.x = randint(2, 9) * 44
                self.apple.y = randint(2, 9) * 44
                self.player.length = self.player.length + 1
                if i == 0:
                    self.score = self.score + 1

        # does snake collide with itself?
        for i in range(2, self.player.length):
            if self.game.isCollision(
                self.player.x[0],
                self.player.y[0],
                self.player.x[i],
                self.player.y[i],
                40,
            ):
                print("You lose! Collision: ")
                print(
                    "x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")"
                )
                print(
                    "x["
                    + str(i)
                    + "] ("
                    + str(self.player.x[i])
                    + ","
                    + str(self.player.y[i])
                    + ")"
                )
                exit(0)

        pass

    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        self.player.draw(self._display_surf, self._image_surf, self._image_dest_rect)
        self.apple.draw(self._display_surf, self._apple_surf, self._apple_dest_rect)
        pygame.transform.flip(self._display_surf, False, False)
        pygame.display.set_caption(f"Score: {self.score}")
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            # Human input
            pygame.event.pump()

            current_direction = self.player.direction
            keys = pygame.key.get_pressed()

            if keys[K_RIGHT]:
                self.player.moveRight()

            if keys[K_LEFT]:
                self.player.moveLeft()

            if keys[K_UP]:
                self.player.moveUp()

            if keys[K_DOWN]:
                self.player.moveDown()

            if keys[K_ESCAPE]:
                self._running = False


            # AI input
            self.agent.act(self.game_state)
            if self.agent.action["K_RIGHT"]:
                self.player.moveRight()

            if self.agent.action["K_LEFT"]:
                self.player.moveLeft()

            if self.agent.action["K_UP"]:
                self.player.moveUp()

            if self.agent.action["K_DOWN"]:
                self.player.moveDown()

            self.log_game_state.add_state(self.game_state, score=self.score, current_direction=current_direction)
            
            self.on_loop()
            self.on_render()

            time.sleep(50.0 / 1000.0)

        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
