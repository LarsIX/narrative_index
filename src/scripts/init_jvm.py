"""
start_jvm()

Initializes the Java Virtual Machine (JVM) for use with IDTxl's Kraskov estimator,
which depends on the Java Information Dynamics Toolkit (JIDT).

This function:
- Starts the JVM only if it’s not already running
- Loads `infodynamics.jar` from the local project structure
- Uses a fixed path to the working `jvm.dll` from JDK 24
  (update if your JDK location changes)

Requirements:
- infodynamics.jar at: <project_root>/IDTxl/idtxl/infodynamics.jar
- jvm.dll at: C:/Program Files/Java/jdk-24/bin/server/jvm.dll
"""

import jpype
from pathlib import Path

def start_jvm():
    if jpype.isJVMStarted():
        print("JVM already running.")
        return

    # Resolve path to infodynamics.jar (JIDT library)
    this_dir = Path(__file__).resolve().parent
    jar_path = (this_dir / "../../IDTxl/idtxl/infodynamics.jar").resolve()

    # Path to working jvm.dll (from your installed JDK 24)
    jvm_path = Path("C:/Program Files/Java/jdk-24/bin/server/jvm.dll")

    # Validate paths
    if not jar_path.exists():
        raise FileNotFoundError(f"infodynamics.jar not found at: {jar_path}")
    if not jvm_path.exists():
        raise FileNotFoundError(f"jvm.dll not found at: {jvm_path}")

    # Start JVM with JIDT classpath
    jpype.startJVM(str(jvm_path), "-ea", f"-Djava.class.path={str(jar_path)}")
    print(f"JVM started with infodynamics.jar at: {jar_path}")
