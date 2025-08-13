
from utils import *
from root2csv import root2csv, split_csv
from process_hit.backward_process import split_hits
from EvalTrack.BackwardEval import BackwardEval
from Check_Eff import Check_Efficiency, Plot_Efficiency




def main():
    if not os.listdir(os.path.join(workdir, "RawData")):
        rprint("FATAL ERROR -- No data found in RawData, Please put in", end="")
        yprint(" RawFile ", end="")
        rprint("and run the code again.")
        exit(0)

    # print(f"Smear X = {SMEAR_N_X}, Y = {SMEAR_N_Y}")
    
    root2csv.main()
    split_csv.main()
    split_hits.main()
    BackwardEval.main()
    Check_Efficiency.main()
    Plot_Efficiency.main()











    pass


if __name__ == "__main__":
    # 在代码中直接调用
    main()
    # print(timer.func_time)
    timer.get_stats()