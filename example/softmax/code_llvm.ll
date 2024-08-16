; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@global_smem = external addrspace(3) global [0 x i8], align 16

define void @softmax_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2, i32 %3, i32 %4) local_unnamed_addr !dbg !7 {
  %6 = tail call i32 asm "mov.u32 $0, %ctaid.x;", "=r"() #3, !dbg !10
  %7 = mul i32 %6, %2, !dbg !11
  %8 = sext i32 %7 to i64, !dbg !12
  %9 = getelementptr float, ptr addrspace(1) %1, i64 %8, !dbg !12
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !13
  %11 = and i32 %10, 31, !dbg !13
  %12 = lshr i32 %10, 5, !dbg !13
  %13 = and i32 %10, 127, !dbg !13
  %14 = or disjoint i32 %13, 128, !dbg !13
  %15 = or disjoint i32 %13, 256, !dbg !13
  %16 = or disjoint i32 %13, 384, !dbg !13
  %17 = or disjoint i32 %13, 512, !dbg !13
  %18 = or disjoint i32 %13, 640, !dbg !13
  %19 = or disjoint i32 %13, 768, !dbg !13
  %20 = or disjoint i32 %13, 896, !dbg !13
  %21 = zext nneg i32 %13 to i64, !dbg !14
  %22 = getelementptr float, ptr addrspace(1) %9, i64 %21, !dbg !14
  %23 = zext nneg i32 %14 to i64, !dbg !14
  %24 = getelementptr float, ptr addrspace(1) %9, i64 %23, !dbg !14
  %25 = zext nneg i32 %15 to i64, !dbg !14
  %26 = getelementptr float, ptr addrspace(1) %9, i64 %25, !dbg !14
  %27 = zext nneg i32 %16 to i64, !dbg !14
  %28 = getelementptr float, ptr addrspace(1) %9, i64 %27, !dbg !14
  %29 = zext nneg i32 %17 to i64, !dbg !14
  %30 = getelementptr float, ptr addrspace(1) %9, i64 %29, !dbg !14
  %31 = zext nneg i32 %18 to i64, !dbg !14
  %32 = getelementptr float, ptr addrspace(1) %9, i64 %31, !dbg !14
  %33 = zext nneg i32 %19 to i64, !dbg !14
  %34 = getelementptr float, ptr addrspace(1) %9, i64 %33, !dbg !14
  %35 = zext nneg i32 %20 to i64, !dbg !14
  %36 = getelementptr float, ptr addrspace(1) %9, i64 %35, !dbg !14
  %37 = icmp slt i32 %13, %4, !dbg !15
  %38 = icmp slt i32 %14, %4, !dbg !15
  %39 = icmp slt i32 %15, %4, !dbg !15
  %40 = icmp slt i32 %16, %4, !dbg !15
  %41 = icmp slt i32 %17, %4, !dbg !15
  %42 = icmp slt i32 %18, %4, !dbg !15
  %43 = icmp slt i32 %19, %4, !dbg !15
  %44 = icmp slt i32 %20, %4, !dbg !15
  %45 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];\0A\09@!$4 mov.u32 $0, $3;", "=r,l,b,r,b"(ptr addrspace(1) %22, i1 %37, i32 -8388608, i1 %37) #3, !dbg !16
  %46 = bitcast i32 %45 to float, !dbg !16
  %47 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];\0A\09@!$4 mov.u32 $0, $3;", "=r,l,b,r,b"(ptr addrspace(1) %24, i1 %38, i32 -8388608, i1 %38) #3, !dbg !16
  %48 = bitcast i32 %47 to float, !dbg !16
  %49 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];\0A\09@!$4 mov.u32 $0, $3;", "=r,l,b,r,b"(ptr addrspace(1) %26, i1 %39, i32 -8388608, i1 %39) #3, !dbg !16
  %50 = bitcast i32 %49 to float, !dbg !16
  %51 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];\0A\09@!$4 mov.u32 $0, $3;", "=r,l,b,r,b"(ptr addrspace(1) %28, i1 %40, i32 -8388608, i1 %40) #3, !dbg !16
  %52 = bitcast i32 %51 to float, !dbg !16
  %53 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];\0A\09@!$4 mov.u32 $0, $3;", "=r,l,b,r,b"(ptr addrspace(1) %30, i1 %41, i32 -8388608, i1 %41) #3, !dbg !16
  %54 = bitcast i32 %53 to float, !dbg !16
  %55 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];\0A\09@!$4 mov.u32 $0, $3;", "=r,l,b,r,b"(ptr addrspace(1) %32, i1 %42, i32 -8388608, i1 %42) #3, !dbg !16
  %56 = bitcast i32 %55 to float, !dbg !16
  %57 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];\0A\09@!$4 mov.u32 $0, $3;", "=r,l,b,r,b"(ptr addrspace(1) %34, i1 %43, i32 -8388608, i1 %43) #3, !dbg !16
  %58 = bitcast i32 %57 to float, !dbg !16
  %59 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];\0A\09@!$4 mov.u32 $0, $3;", "=r,l,b,r,b"(ptr addrspace(1) %36, i1 %44, i32 -8388608, i1 %44) #3, !dbg !16
  %60 = bitcast i32 %59 to float, !dbg !16
  %61 = tail call float @llvm.maxnum.f32(float %46, float %48), !dbg !17
  %62 = tail call float @llvm.maxnum.f32(float %61, float %50), !dbg !17
  %63 = tail call float @llvm.maxnum.f32(float %62, float %52), !dbg !17
  %64 = tail call float @llvm.maxnum.f32(float %63, float %54), !dbg !17
  %65 = tail call float @llvm.maxnum.f32(float %64, float %56), !dbg !17
  %66 = tail call float @llvm.maxnum.f32(float %65, float %58), !dbg !17
  %67 = tail call float @llvm.maxnum.f32(float %66, float %60), !dbg !17
  %68 = bitcast float %67 to i32, !dbg !22
  %69 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %68, i32 16, i32 31), !dbg !22
  %70 = bitcast i32 %69 to float, !dbg !22
  %71 = tail call float @llvm.maxnum.f32(float %67, float %70), !dbg !17
  %72 = bitcast float %71 to i32, !dbg !22
  %73 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %72, i32 8, i32 31), !dbg !22
  %74 = bitcast i32 %73 to float, !dbg !22
  %75 = tail call float @llvm.maxnum.f32(float %71, float %74), !dbg !17
  %76 = bitcast float %75 to i32, !dbg !22
  %77 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %76, i32 4, i32 31), !dbg !22
  %78 = bitcast i32 %77 to float, !dbg !22
  %79 = tail call float @llvm.maxnum.f32(float %75, float %78), !dbg !17
  %80 = bitcast float %79 to i32, !dbg !22
  %81 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %80, i32 2, i32 31), !dbg !22
  %82 = bitcast i32 %81 to float, !dbg !22
  %83 = tail call float @llvm.maxnum.f32(float %79, float %82), !dbg !17
  %84 = bitcast float %83 to i32, !dbg !22
  %85 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %84, i32 1, i32 31), !dbg !22
  %86 = bitcast i32 %85 to float, !dbg !22
  %87 = tail call float @llvm.maxnum.f32(float %83, float %86), !dbg !17
  %88 = icmp eq i32 %11, 0, !dbg !22
  %89 = and i32 %12, 3, !dbg !22
  %90 = zext nneg i32 %89 to i64, !dbg !22
  %91 = getelementptr float, ptr addrspace(3) @global_smem, i64 %90, !dbg !22
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %91, float %87, i1 %88) #3, !dbg !22
  tail call void @llvm.nvvm.barrier0(), !dbg !22
  %92 = icmp slt i32 %10, 4, !dbg !22
  %93 = sext i32 %10 to i64, !dbg !22
  %94 = getelementptr float, ptr addrspace(3) @global_smem, i64 %93, !dbg !22
  %95 = tail call float asm sideeffect "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b"(ptr addrspace(3) %94, i1 %92) #3, !dbg !22
  %96 = bitcast float %95 to i32, !dbg !22
  %97 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %96, i32 2, i32 31), !dbg !22
  %98 = bitcast i32 %97 to float, !dbg !22
  %99 = tail call float @llvm.maxnum.f32(float %95, float %98), !dbg !17
  %100 = bitcast float %99 to i32, !dbg !22
  %101 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %100, i32 1, i32 31), !dbg !22
  %102 = bitcast i32 %101 to float, !dbg !22
  %103 = tail call float @llvm.maxnum.f32(float %99, float %102), !dbg !17
  %104 = and i32 %10, 3, !dbg !22
  %105 = icmp eq i32 %104, 0, !dbg !22
  %106 = and i1 %92, %105, !dbg !22
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %94, float %103, i1 %106) #3, !dbg !22
  tail call void @llvm.nvvm.barrier0(), !dbg !22
  %107 = load float, ptr addrspace(3) @global_smem, align 16, !dbg !22
  %108 = fsub float %46, %107, !dbg !23
  %109 = fsub float %48, %107, !dbg !23
  %110 = fsub float %50, %107, !dbg !23
  %111 = fsub float %52, %107, !dbg !23
  %112 = fsub float %54, %107, !dbg !23
  %113 = fsub float %56, %107, !dbg !23
  %114 = fsub float %58, %107, !dbg !23
  %115 = fsub float %60, %107, !dbg !23
  %116 = fmul float %108, 0x3FF7154760000000, !dbg !24
  %117 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %116) #3, !dbg !24
  %118 = fmul float %109, 0x3FF7154760000000, !dbg !24
  %119 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %118) #3, !dbg !24
  %120 = fmul float %110, 0x3FF7154760000000, !dbg !24
  %121 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %120) #3, !dbg !24
  %122 = fmul float %111, 0x3FF7154760000000, !dbg !24
  %123 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %122) #3, !dbg !24
  %124 = fmul float %112, 0x3FF7154760000000, !dbg !24
  %125 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %124) #3, !dbg !24
  %126 = fmul float %113, 0x3FF7154760000000, !dbg !24
  %127 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %126) #3, !dbg !24
  %128 = fmul float %114, 0x3FF7154760000000, !dbg !24
  %129 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %128) #3, !dbg !24
  %130 = fmul float %115, 0x3FF7154760000000, !dbg !24
  %131 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %130) #3, !dbg !24
  tail call void @llvm.nvvm.barrier0(), !dbg !25
  %132 = fadd float %117, %119, !dbg !27
  %133 = fadd float %132, %121, !dbg !27
  %134 = fadd float %133, %123, !dbg !27
  %135 = fadd float %134, %125, !dbg !27
  %136 = fadd float %135, %127, !dbg !27
  %137 = fadd float %136, %129, !dbg !27
  %138 = fadd float %137, %131, !dbg !27
  %139 = bitcast float %138 to i32, !dbg !25
  %140 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %139, i32 16, i32 31), !dbg !25
  %141 = bitcast i32 %140 to float, !dbg !25
  %142 = fadd float %138, %141, !dbg !27
  %143 = bitcast float %142 to i32, !dbg !25
  %144 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %143, i32 8, i32 31), !dbg !25
  %145 = bitcast i32 %144 to float, !dbg !25
  %146 = fadd float %142, %145, !dbg !27
  %147 = bitcast float %146 to i32, !dbg !25
  %148 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %147, i32 4, i32 31), !dbg !25
  %149 = bitcast i32 %148 to float, !dbg !25
  %150 = fadd float %146, %149, !dbg !27
  %151 = bitcast float %150 to i32, !dbg !25
  %152 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %151, i32 2, i32 31), !dbg !25
  %153 = bitcast i32 %152 to float, !dbg !25
  %154 = fadd float %150, %153, !dbg !27
  %155 = bitcast float %154 to i32, !dbg !25
  %156 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %155, i32 1, i32 31), !dbg !25
  %157 = bitcast i32 %156 to float, !dbg !25
  %158 = fadd float %154, %157, !dbg !27
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %91, float %158, i1 %88) #3, !dbg !25
  tail call void @llvm.nvvm.barrier0(), !dbg !25
  %159 = tail call float asm sideeffect "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b"(ptr addrspace(3) %94, i1 %92) #3, !dbg !25
  %160 = bitcast float %159 to i32, !dbg !25
  %161 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %160, i32 2, i32 31), !dbg !25
  %162 = bitcast i32 %161 to float, !dbg !25
  %163 = fadd float %159, %162, !dbg !27
  %164 = bitcast float %163 to i32, !dbg !25
  %165 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %164, i32 1, i32 31), !dbg !25
  %166 = bitcast i32 %165 to float, !dbg !25
  %167 = fadd float %163, %166, !dbg !27
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %94, float %167, i1 %106) #3, !dbg !25
  tail call void @llvm.nvvm.barrier0(), !dbg !25
  %168 = load float, ptr addrspace(3) @global_smem, align 16, !dbg !25
  %169 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %117, float %168) #3, !dbg !28
  %170 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %119, float %168) #3, !dbg !28
  %171 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %121, float %168) #3, !dbg !28
  %172 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %123, float %168) #3, !dbg !28
  %173 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %125, float %168) #3, !dbg !28
  %174 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %127, float %168) #3, !dbg !28
  %175 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %129, float %168) #3, !dbg !28
  %176 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %131, float %168) #3, !dbg !28
  %177 = mul i32 %6, %3, !dbg !29
  %178 = sext i32 %177 to i64, !dbg !30
  %179 = getelementptr float, ptr addrspace(1) %0, i64 %178, !dbg !30
  %180 = getelementptr float, ptr addrspace(1) %179, i64 %21, !dbg !31
  %181 = getelementptr float, ptr addrspace(1) %179, i64 %23, !dbg !31
  %182 = getelementptr float, ptr addrspace(1) %179, i64 %25, !dbg !31
  %183 = getelementptr float, ptr addrspace(1) %179, i64 %27, !dbg !31
  %184 = getelementptr float, ptr addrspace(1) %179, i64 %29, !dbg !31
  %185 = getelementptr float, ptr addrspace(1) %179, i64 %31, !dbg !31
  %186 = getelementptr float, ptr addrspace(1) %179, i64 %33, !dbg !31
  %187 = getelementptr float, ptr addrspace(1) %179, i64 %35, !dbg !31
  %188 = bitcast float %169 to i32, !dbg !32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %188, ptr addrspace(1) %180, i1 %37) #3, !dbg !32
  %189 = bitcast float %170 to i32, !dbg !32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %189, ptr addrspace(1) %181, i1 %38) #3, !dbg !32
  %190 = bitcast float %171 to i32, !dbg !32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %190, ptr addrspace(1) %182, i1 %39) #3, !dbg !32
  %191 = bitcast float %172 to i32, !dbg !32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %191, ptr addrspace(1) %183, i1 %40) #3, !dbg !32
  %192 = bitcast float %173 to i32, !dbg !32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %192, ptr addrspace(1) %184, i1 %41) #3, !dbg !32
  %193 = bitcast float %174 to i32, !dbg !32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %193, ptr addrspace(1) %185, i1 %42) #3, !dbg !32
  %194 = bitcast float %175 to i32, !dbg !32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %194, ptr addrspace(1) %186, i1 %43) #3, !dbg !32
  %195 = bitcast float %176 to i32, !dbg !32
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %195, ptr addrspace(1) %187, i1 %44) #3, !dbg !32
  ret void, !dbg !33
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #2

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!nvvm.annotations = !{!4, !5}
!llvm.ident = !{!6}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!3 = !DIFile(filename: "test_softmax.py", directory: "/home/yongqi/external/triton_shared/python/examples")
!4 = !{ptr @softmax_kernel, !"kernel", i32 1}
!5 = !{ptr @softmax_kernel, !"maxntidx", i32 128}
!6 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!7 = distinct !DISubprogram(name: "softmax_kernel", linkageName: "softmax_kernel", scope: !3, file: !3, line: 11, type: !8, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!8 = !DISubroutineType(cc: DW_CC_normal, types: !9)
!9 = !{}
!10 = !DILocation(line: 13, column: 28, scope: !7)
!11 = !DILocation(line: 15, column: 42, scope: !7)
!12 = !DILocation(line: 15, column: 32, scope: !7)
!13 = !DILocation(line: 18, column: 31, scope: !7)
!14 = !DILocation(line: 19, column: 33, scope: !7)
!15 = !DILocation(line: 21, column: 49, scope: !7)
!16 = !DILocation(line: 21, column: 18, scope: !7)
!17 = !DILocation(line: 164, column: 27, scope: !18, inlinedAt: !21)
!18 = distinct !DILexicalBlockFile(scope: !20, file: !19, discriminator: 0)
!19 = !DIFile(filename: "standard.py", directory: "/home/yongqi/.local/lib/python3.12/site-packages/triton/language")
!20 = distinct !DILexicalBlockFile(scope: !7, file: !19, discriminator: 0)
!21 = !DILocation(line: 23, column: 33, scope: !7)
!22 = !DILocation(line: 185, column: 40, scope: !20, inlinedAt: !21)
!23 = !DILocation(line: 23, column: 26, scope: !7)
!24 = !DILocation(line: 25, column: 23, scope: !7)
!25 = !DILocation(line: 268, column: 36, scope: !20, inlinedAt: !26)
!26 = !DILocation(line: 26, column: 25, scope: !7)
!27 = !DILocation(line: 257, column: 15, scope: !18, inlinedAt: !26)
!28 = !DILocation(line: 27, column: 33, scope: !7)
!29 = !DILocation(line: 29, column: 50, scope: !7)
!30 = !DILocation(line: 29, column: 40, scope: !7)
!31 = !DILocation(line: 30, column: 41, scope: !7)
!32 = !DILocation(line: 31, column: 26, scope: !7)
!33 = !DILocation(line: 31, column: 4, scope: !7)
