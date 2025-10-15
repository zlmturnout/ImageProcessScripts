
import pygame
import random
import math
import sys

# 初始化pygame
pygame.init()

# 设置窗口
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("带呼吸效果的数码水母")

# 颜色定义
BLUE = (0, 105, 148)
LIGHT_BLUE = (100, 200, 255)
PURPLE = (138, 43, 226)
PINK = (255, 105, 180)

class Jellyfish:
    def __init__(self):
        self.base_radius = random.randint(30, 60)
        self.x = random.randint(self.base_radius, WIDTH - self.base_radius)
        self.y = random.randint(self.base_radius, HEIGHT - self.base_radius)
        self.speed = random.uniform(0.5, 2.0)
        self.direction = random.uniform(0, 2 * math.pi)
        self.tentacles = random.randint(6, 12)
        self.color = random.choice([LIGHT_BLUE, PURPLE, PINK])
        self.breath_speed = random.uniform(0.05, 0.1)
        self.breath_phase = 0
        self.breath_amplitude = random.uniform(0.1, 0.3)  # 呼吸幅度
        
    def move(self):
        # 随机改变方向
        if random.random() < 0.02:
            self.direction += random.uniform(-0.5, 0.5)
            
        # 边界检测
        if self.x < self.base_radius or self.x > WIDTH - self.base_radius:
            self.direction = math.pi - self.direction
        if self.y < self.base_radius or self.y > HEIGHT - self.base_radius:
            self.direction = -self.direction
            
        # 移动
        self.x += math.cos(self.direction) * self.speed
        self.y += math.sin(self.direction) * self.speed
        
        # 呼吸效果 - 使用正弦函数模拟自然呼吸
        self.breath_phase += self.breath_speed
        self.current_radius = self.base_radius * (1 + self.breath_amplitude * math.sin(self.breath_phase))
        
    def draw(self, surface):
        # 绘制水母主体
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.current_radius))
        
        # 绘制触手
        for i in range(self.tentacles):
            angle = 2 * math.pi * i / self.tentacles
            end_x = self.x + math.cos(angle) * (self.current_radius + 30)
            end_y = self.y + math.sin(angle) * (self.current_radius + 30)
            pygame.draw.line(surface, self.color, (self.x, self.y), (end_x, end_y), 2)
            
        # 绘制眼睛
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x - 10), int(self.y - 5)), 5)
        pygame.draw.circle(surface, (0, 0, 0), (int(self.x - 10), int(self.y - 5)), 2)
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x + 10), int(self.y - 5)), 5)
        pygame.draw.circle(surface, (0, 0, 0), (int(self.x + 10), int(self.y - 5)), 2)

def main():
    clock = pygame.time.Clock()
    jellyfishes = [Jellyfish() for _ in range(5)]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        screen.fill(BLUE)
        
        for jellyfish in jellyfishes:
            jellyfish.move()
            jellyfish.draw(screen)
            
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
