import time
from functools import wraps

class FunctionTimer:
    def __init__(self, verbose=False):
        self.verbose = verbose  # 是否在每次调用后打印详情
        self.reset_stats()
        self.func_time = {}

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            # 记录开始时间
            start_time = time.perf_counter()

            # 执行原函数
            result = func(*args, **kwargs)

            # 计算耗时
            duration = time.perf_counter() - start_time



            try:
                self.func_time[func.__name__].append(duration)
            except KeyError:
                self.func_time[func.__name__] = [duration]


            # 如果需要立即输出
            if self.verbose:
                print(f"[{func.__name__}] 第{self.count}次调用 | 本次耗时: {duration:.6f}s")

            return result

        return wrapped

    def reset_stats(self):
        """重置统计信息"""
        self.count = 0
        self.total_time = 0.0
        self.times = []
        self.min_time = float('inf')
        self.max_time = 0.0

    def get_stats(self):
        """获取统计报告"""
        # if self.count == 0:
        #     return "暂无调用记录"

        for func, times in self.func_time.items():
            print(f"{func} 平均耗时: {sum(times) / len(times):.6f}s, 调用次数: {len(times)}，总耗时: {sum(times):.6f}s")

timer = FunctionTimer()

# # 使用示例
# if __name__ == "__main__":
#     # 初始化装饰器（verbose=True 表示每次调用后打印详情）
#     timer = FunctionTimer(verbose=True)
#
#
#     # 应用装饰器
#     @timer
#     def example_function(n):
#         time.sleep(n * 0.1)
#         return n * 2
#
#     @timer
#     def example_function2(n):
#         time.sleep(n * 0.1)
#         return n * 2
#
#
#
#     # 测试调用
#     example_function(1)
#     example_function(2)
#     example_function(0.5)
#     example_function2(1.5)
#     example_function2(2.5)
#
#     # 打印统计报告
#     # print("\n" + timer.get_stats())
#
#     # 重置统计
#     timer.reset_stats()
#     print("\n重置后统计:", timer.get_stats())