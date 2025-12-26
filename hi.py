from triton.backends.triton_tvm.driver import TVMDriver
triton.runtime.driver.set_active(TVMDriver())