@echo off
rem Environment configuration
set anaconda_path=C:/Software/Anaconda3/Scripts/
set sourcecode_path=C:/SourceCode/liam-bui
set model_path=%sourcecode_path%/best_model/best_model.h5

rem Program Execution
call %anaconda_path%/activate.bat
cd %sourcecode_path%
python pipelines/classification_pipeline.py --data_path="%1" --model_path=%model_path% --img_size=299 --batch_size=16
call %anaconda_path%/deactivate.bat
pause >nul|(echo Press any key to end the program)