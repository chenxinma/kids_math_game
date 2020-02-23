# /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import random
import pygame
from pygame.locals import \
    K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9
from pygame.color import THECOLORS

import pymunk
# from pymunk import Vec2d
import pymunk.pygame_util

BALL_COLOR = (80, 80, 80)
BALL_COLOR_ACTIVITY = (255, 179, 25)
BALL_SIZE_RATE = 2.0

# BACKGROUND = pygame.image.load('flapBG.png')
# BACKGROUND = pygame.transform.scale(BACKGROUND, (800, 600))
BACKGROUND = pygame.Surface([1280, 720])
BACKGROUND.fill(THECOLORS['white'])
BACKGROUND_RECT = BACKGROUND.get_rect()


def flipy(y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    return -y + 720


class GLabel(pygame.sprite.Sprite):
    position = [0, 0]
    align = 'center'

    def __init__(self,
                 font,
                 text="[Blank]",
                 color=(93, 138, 168),
                 position=[0, 0],
                 align='center'):
        super().__init__()
        self.font = font
        self.color = color
        self.position = position
        self.align = align
        self.set_text(text)

    def set_text(self, text):
        self.text = text
        img_text = self.font.render(self.text, 1, self.color)
        tw, th = img_text.get_rect().size
        self.image = img_text
        if 'center' == self.align:
            self.rect = pygame.Rect(self.position[0] - tw / 2,
                                    flipy(self.position[1] - th / 2), tw, th)
        elif 'left' == self.align:
            self.rect = pygame.Rect(self.position[0],
                                    flipy(self.position[1] - th / 2), tw, th)
        elif 'right' == self.align:
            self.rect = pygame.Rect(self.position[0] - tw,
                                    flipy(self.position[1] - th / 2), tw, th)


class FormulaLabel(GLabel):
    def __init__(self, font, text='[Blank]', g_holder=None):
        super().__init__(font, text=text, position=[640, 250])
        self.MAX_STACK = 2
        self.input_stack = []
        self.operate = ""
        self.result = None
        self.g_holder = g_holder

    def set_answer(self, result):
        self.result = result
        self.input_stack.clear()
        self.flush_operate()
        self.flush_text()

    def flush_operate(self):
        self.operate = random.choice(["+", "-"])
        if self.result > 8 and self.operate == '-':
            self.operate = '+'
        if self.result == 1 and self.operate == '+':
            self.operate = '-'

    def flush_text(self):
        t = "Ready!"
        if len(self.input_stack) == 1:
            t = "%s %s ? = %d" % (self.input_stack[0], self.operate,
                                  self.result)
        elif len(self.input_stack) == 2:
            t = "%s %s %s = %d" % (self.input_stack[0], self.operate,
                                   self.input_stack[1], self.result)
        else:
            t = "? %s ? = %d" % (self.operate, self.result)
        self.set_text(t)

    def put(self, num):
        if self.MAX_STACK > len(self.input_stack):
            self.input_stack.append(num)
        self.flush_text()

    def valify(self):
        if self.MAX_STACK == len(self.input_stack):
            f = "".join(
                [self.input_stack[0], self.operate, self.input_stack[1]])
            v = eval(f)
            self.input_stack.clear()
            return v == self.result
        return False

    def update(self, keys):
        self.handle_input(keys)

    def handle_input(self, keys):
        if keys in [K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9]:
            self.put(str(keys - K_1 + 1))


class NumBall(pygame.sprite.Sprite):
    def __init__(self, shape, ball_font, num=0):
        super().__init__()
        self.shape = shape
        self.num = num
        self.activity = False
        self.ball_font = ball_font
        r = self.shape.radius
        self.image = pygame.Surface([r * 2, r * 2], pygame.SRCALPHA)
        self.update_image()

    def update_image(self):
        r = self.shape.radius
        v = self.shape.body.position
        rot = self.shape.body.rotation_vector
        # p2 = Vec2d(rot.x, -rot.y) * r * 0.9
        pygame.draw.circle(
            self.image, (self.activity and BALL_COLOR_ACTIVITY or BALL_COLOR),
            (int(r), int(r)), int(r), 0)
        ball_text = self.ball_font.render(str(self.num), 1, THECOLORS["white"])
        rball_text = pygame.transform.rotate(ball_text, rot.angle_degrees)
        tw, th = rball_text.get_rect().size
        self.image.blit(rball_text, (r - tw / 2, r - th / 2))

        self.rect = pygame.Rect(int(v.x - r), int(flipy(v.y) - r), int(r * 2),
                                int(r * 2))

    def update(self):
        self.update_image()

    @staticmethod
    def create(space, ball_font, num=1):
        mass = 0.1
        radius = 20 + num * 2
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = random.randint(320, 960)
        body.position = x, 620
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.friction = 0.5
        space.add(body, shape)

        return NumBall(shape, ball_font, num)

    @staticmethod
    def get_activity_ball(balls):
        # Only one ball
        if len(balls) == 0:
            return None
        # Only one ball
        elif len(balls) == 1:
            balls[0].activity = True
            return balls[0]

        act_balls = [b for b in balls if b.activity]
        if len(act_balls) == 0:
            # choice a ball
            idx = random.randint(0, len(balls) - 1)
            balls[idx].activity = True
            return balls[idx]
        else:
            # exists active ball
            return act_balls[0]


class NumBallGroup(pygame.sprite.Group):
    def __init__(self, g_holder):
        super().__init__()
        self.MAX_BALLS = 20

        self.ticks_to_next_ball = 10
        self.balls_to_remove = []
        self.active_ball = None
        self.g_holder = g_holder
        self.stage = 0
        self.load_ball_count = 0

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        # ball generate & new formula
        self.ticks_to_next_ball -= 1
        if self.stage == 0 \
                and self.ticks_to_next_ball <= 0:
            self.ticks_to_next_ball = 30
            self.g_holder.create_ball(self)
            self.load_ball_count += 1

        if self.load_ball_count == self.MAX_BALLS:
            self.stage = 1
            self.load_ball_count = 0

        if len(self.sprites()) == 0:
            self.stage = 0

        for ball in self.sprites():
            if ball.shape.body.position.y < 200:
                self.balls_to_remove.append(ball)
        for ball in self.balls_to_remove:
            self.g_holder.space.remove(ball.shape, ball.shape.body)
            self.remove(ball)
        self.balls_to_remove.clear()

        if self.active_ball is None:
            self.active_ball = NumBall.get_activity_ball(self.sprites())
            if self.active_ball:
                self.g_holder.formula.set_answer(self.active_ball.num)

    def remove_active_ball(self):
        self.balls_to_remove.append(self.active_ball)
        self.active_ball = None


class Game(object):
    """Controls entire game"""
    def __init__(self):
        self.CALC_RANGE = 18

        self.screen = self.setup_pygame()
        self.screen_rect = self.screen.get_rect()
        self.label_group = pygame.sprite.Group()
        self.ball_group = NumBallGroup(self)
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.done = False
        self.present_time = 0.0
        self.score = 0

        self.fonts = {
            "BALL_FONT": pygame.font.Font("res/msyhl.ttc", 32),
            "FORMUL_FONT": pygame.font.Font("res/msyhbd.ttc", 80),
            "SCORE_FONT": pygame.font.Font("res/7px2bus.ttf", 18)
        }
        self.fonts['BALL_FONT'].set_underline(True)
        self.formula = self.create_formula(self.label_group)

        self.space = self.setup_physics()
        self.score_label = GLabel(font=self.fonts['SCORE_FONT'],
                                  text='score = 0',
                                  color=THECOLORS['red'],
                                  position=[1258, 64],
                                  align='right')
        self.label_group.add(self.score_label)

    def setup_physics(self):
        # Physics stuff
        space = pymunk.Space()
        space.gravity = (0.0, -900.0)
        # draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        # # disable the build in debug draw of collision
        # #   point since we use our own code.
        # draw_options.flags = \
        #     draw_options.flags ^ \
        #     pymunk.pygame_util.DrawOptions.DRAW_COLLISION_POINTS

        # walls
        static_lines = [
            pymunk.Segment(space.static_body, (18.0, 275.0), (18.0, 702.0),
                           0.0),
            pymunk.Segment(space.static_body, (18.0, 702.0), (1262.0, 702.0),
                           0.0),
            pymunk.Segment(space.static_body, (1262.0, 702.0), (1262.0, 360.0),
                           0.0),
            pymunk.Segment(space.static_body, (1262.0, 360.0), (18.0, 275.0),
                           0.0)
        ]
        for l in static_lines:
            l.friction = 0.5
        space.add(static_lines)

        # draw wall
        for line in static_lines:
            body = line.body

            pv1 = body.position + line.a.rotated(body.angle)
            pv2 = body.position + line.b.rotated(body.angle)
            p1 = pv1.x, flipy(pv1.y)
            p2 = pv2.x, flipy(pv2.y)
            pygame.draw.lines(BACKGROUND, THECOLORS["lightgray"], False,
                              [p1, p2], 2)

        # ch = space.add_collision_handler(0, 0)
        # ch.data["surface"] = self.screen

        return space

    def setup_pygame(self):
        """Initializes pygame and produces a surface to blit on"""
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.init()
        pygame.display.set_caption('Arithmetic Game')
        screen = pygame.display.set_mode((1280, 720))

        return screen

    def create_formula(self, label_group):
        """Creates a formula to control"""
        f = FormulaLabel(self.fonts['FORMUL_FONT'], "Ready!", self)
        label_group.add(f)

        return f

    def create_ball(self, ball_group):
        num = random.randint(1, self.CALC_RANGE)
        ball = NumBall.create(self.space, self.fonts['BALL_FONT'], num)
        ball_group.add(ball)
        return ball

    def update(self):
        """Updates entire game"""
        while not self.done:
            if self.formula.valify():
                self.score += 100
                self.score_label.set_text('score = %d' % self.score)
                self.ball_group.remove_active_ball()
            # input
            self.keys = self.get_user_input()
            # update
            self.label_group.update(self.keys)
            self.ball_group.update()
            # Update physics
            dt = (pygame.time.get_ticks() - self.present_time) / 1000.0
            self.space.step(dt)
            self.present_time = pygame.time.get_ticks()
            # draw
            self.screen.blit(BACKGROUND, BACKGROUND_RECT)
            self.ball_group.draw(self.screen)
            self.label_group.draw(self.screen)

            # pygame.display.update()
            pygame.display.flip()
            self.clock.tick(self.fps)
            pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

    def get_user_input(self):
        """Get's user events and keys pressed"""
        key = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key == pygame.K_ESCAPE):
                self.done = True
            if event.type == pygame.KEYDOWN:
                key = event.key

        # keys = pygame.key.get_pressed()
        return key


if __name__ == '__main__':
    game = Game()
    game.update()
    pygame.quit()
    sys.exit()
