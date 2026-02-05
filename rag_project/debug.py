import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Current Working Directory: {os.getcwd()}")

print("\n'langchain' 모듈 위치 찾는 중...")
try:
    import langchain
    print(f"langchain found at: {langchain.__file__}")
    print(f"langchain version: {langchain.__version__}")
except ImportError:
    print("langchain 모듈 자체를 찾을 수 없습니다.")
except AttributeError:
    # langchain이 파일이 아니라 폴더(Namespace)로 인식될 때
    print(f"langchain found (namespace pkg): {langchain.__path__}")

print("\n'langchain.chains' 위치 찾는 중...")
try:
    import langchain.chains
    print(f"langchain.chains found at: {langchain.chains.__file__}")
except ImportError as e:
    print(f"langchain.chains import 실패: {e}")
    
    # 혹시 로컬 파일 충돌인지 확인
    local_shadow = os.path.join(os.getcwd(), "langchain.py")
    local_folder = os.path.join(os.getcwd(), "langchain")
    
    if os.path.exists(local_shadow):
        print(f"\n[발견] 현재 폴더에 '{local_shadow}' 파일이 있습니다.")
        print("이 파일 때문에 파이썬이 진짜 라이브러리를 못 읽습니다. 파일 이름을 바꾸거나 지우세요.")
    elif os.path.exists(local_folder):
        print(f"\n[발견] 현재 폴더에 '{local_folder}' 폴더가 있습니다.")
        print("이 폴더 이름을 바꾸거나 지우세요.")
    else:
        print("\n 로컬 파일 충돌은 아닙니다. 패키지 설치 상태를 의심해야 합니다.")