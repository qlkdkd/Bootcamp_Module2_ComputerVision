for i in range(1, 91):                            # 90장 = 30fps로 3초
    if i % 30 == 0:
        print(f"{i}번째 프레임: {i // 30}초 지남")
