

export GFLAGS_PATH="/public/software/apps/DeepLearning/PyTorch_Lib/gflags-2.1.0-build"
export GLOG_PATH="/public/software/apps/DeepLearning/PyTorch_Lib/glog-build"
export LEVELDB_PATH="/public/software/apps/DeepLearning/PyTorch_Lib/leveldb-1.22-build"
export LMDB_PATH="/public/software/apps/DeepLearning/PyTorch_Lib/lmdb-0.9.24-build"
export OPENBLAS_PATH="/public/software/apps/DeepLearning/PyTorch_Lib/openblas-0.3.7-build"
export OPENCV_PATH="/public/software/apps/DeepLearning/PyTorch_Lib/opencv-2.4.13.6-build"
export OPENMP_PATH="/public/software/apps/DeepLearning/PyTorch_Lib/openmp-build"
export HDIS_PATH="/public/software/apps/DeepLearning/PyTorch_Lib/hiredis-0.12.0-build"

export LD_LIBRARY_PATH=$GFLAGS_PATH/lib:$GLOG_PATH/lib:$LEVELDB_PATH/lib64:$LMDB_PATH/lib:$OPENBLAS_PATH/lib:$OPENCV_PATH/lib:$OPENMP_PATH/lib:$HDIS_PATH/lib:$LD_LIBRARY_PATH
#export CPATH=$GFLAGS_PATH/include:$GLOG_PATH/include:$LEVELDB_PATH/include:$LMDB_PATH/include:$OPENBLAS_PATH/include:$OPENCV_PATH/include:$OPENMP_PATH/include:$HDIS_PATH/include:$CPATH


export C_INCLUDE_PATH=/public/software/apps/DeepLearning/PyTorch_Lib/gflags-2.1.2-build/include:/public/software/apps/DeepLearning/PyTorch_Lib/glog-build/include:/public/software/apps/DeepLearning/PyTorch_Lib/openblas-0.3.7-build/include:/public/software/apps/DeepLearning/PyTorch_Lib/lmdb-0.9.24-build/include:/public/software/apps/DeepLearning/PyTorch_Lib/opencv-2.4.13.6-build/include:/public/software/apps/DeepLearning/PyTorch_Lib/hiredis-0.12.0-build/include:/public/software/apps/DeepLearning/PyTorch_Lib/leveldb-1.22-build/include:/public/software/apps/DeepLearning/PyTorch_Lib/lmdb-0.9.24-build/include:$C_INCLUDE_PATH

export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH
