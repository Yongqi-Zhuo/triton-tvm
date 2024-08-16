; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@global_smem = external addrspace(3) global [0 x i8], align 16

define void @matmul_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8) local_unnamed_addr !dbg !7 {
  %10 = tail call i32 asm "mov.u32 $0, %ctaid.x;", "=r"() #2, !dbg !10
  %11 = add i32 %3, 31, !dbg !11
  %12 = sdiv i32 %11, 32, !dbg !15
  %13 = add i32 %4, 63, !dbg !16
  %14 = sdiv i32 %13, 64, !dbg !18
  %15 = shl nsw i32 %14, 3, !dbg !19
  %.frozen = freeze i32 %10
  %.frozen136 = freeze i32 %15
  %16 = sdiv i32 %.frozen, %.frozen136, !dbg !20
  %17 = shl i32 %16, 3, !dbg !21
  %18 = sub i32 %12, %17, !dbg !22
  %19 = tail call i32 @llvm.smin.i32(i32 %18, i32 8), !dbg !23
  %20 = srem i32 %10, %19, !dbg !24
  %21 = add i32 %17, %20, !dbg !25
  %22 = mul i32 %16, %.frozen136
  %.decomposed = sub i32 %.frozen, %22
  %23 = sdiv i32 %.decomposed, %19, !dbg !26
  %24 = shl i32 %21, 5, !dbg !27
  %25 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !28
  %26 = and i32 %25, 31, !dbg !28
  %27 = lshr i32 %25, 4, !dbg !28
  %28 = and i32 %27, 7, !dbg !28
  %29 = or disjoint i32 %28, 8, !dbg !28
  %30 = or disjoint i32 %28, 16, !dbg !28
  %31 = or disjoint i32 %28, 24, !dbg !28
  %32 = or disjoint i32 %24, %28, !dbg !29
  %33 = or disjoint i32 %24, %29, !dbg !29
  %34 = or disjoint i32 %24, %30, !dbg !29
  %35 = or disjoint i32 %24, %31, !dbg !29
  %36 = srem i32 %32, %3, !dbg !30
  %37 = srem i32 %33, %3, !dbg !30
  %38 = srem i32 %34, %3, !dbg !30
  %39 = srem i32 %35, %3, !dbg !30
  %40 = shl i32 %23, 6, !dbg !31
  %41 = or disjoint i32 %28, 32, !dbg !32
  %42 = or disjoint i32 %28, 40, !dbg !32
  %43 = or disjoint i32 %28, 48, !dbg !32
  %44 = or disjoint i32 %28, 56, !dbg !32
  %45 = and i32 %25, 15, !dbg !32
  %46 = or disjoint i32 %40, %28, !dbg !33
  %47 = or disjoint i32 %40, %29, !dbg !33
  %48 = or disjoint i32 %40, %30, !dbg !33
  %49 = or disjoint i32 %40, %31, !dbg !33
  %50 = or disjoint i32 %40, %41, !dbg !33
  %51 = or disjoint i32 %40, %42, !dbg !33
  %52 = or disjoint i32 %40, %43, !dbg !33
  %53 = or disjoint i32 %40, %44, !dbg !33
  %54 = srem i32 %46, %4, !dbg !34
  %55 = srem i32 %47, %4, !dbg !34
  %56 = srem i32 %48, %4, !dbg !34
  %57 = srem i32 %49, %4, !dbg !34
  %58 = srem i32 %50, %4, !dbg !34
  %59 = srem i32 %51, %4, !dbg !34
  %60 = srem i32 %52, %4, !dbg !34
  %61 = srem i32 %53, %4, !dbg !34
  %62 = mul i32 %36, %6, !dbg !35
  %63 = mul i32 %37, %6, !dbg !35
  %64 = mul i32 %38, %6, !dbg !35
  %65 = mul i32 %39, %6, !dbg !35
  %66 = add i32 %62, %45, !dbg !36
  %67 = add i32 %63, %45, !dbg !36
  %68 = add i32 %64, %45, !dbg !36
  %69 = add i32 %65, %45, !dbg !36
  %70 = sext i32 %66 to i64, !dbg !37
  %71 = getelementptr float, ptr addrspace(1) %0, i64 %70, !dbg !37
  %72 = sext i32 %67 to i64, !dbg !37
  %73 = getelementptr float, ptr addrspace(1) %0, i64 %72, !dbg !37
  %74 = sext i32 %68 to i64, !dbg !37
  %75 = getelementptr float, ptr addrspace(1) %0, i64 %74, !dbg !37
  %76 = sext i32 %69 to i64, !dbg !37
  %77 = getelementptr float, ptr addrspace(1) %0, i64 %76, !dbg !37
  %78 = mul i32 %45, %7, !dbg !38
  %79 = add i32 %54, %78, !dbg !39
  %80 = add i32 %55, %78, !dbg !39
  %81 = add i32 %56, %78, !dbg !39
  %82 = add i32 %57, %78, !dbg !39
  %83 = add i32 %58, %78, !dbg !39
  %84 = add i32 %59, %78, !dbg !39
  %85 = add i32 %60, %78, !dbg !39
  %86 = add i32 %61, %78, !dbg !39
  %87 = sext i32 %79 to i64, !dbg !40
  %88 = getelementptr float, ptr addrspace(1) %1, i64 %87, !dbg !40
  %89 = sext i32 %80 to i64, !dbg !40
  %90 = getelementptr float, ptr addrspace(1) %1, i64 %89, !dbg !40
  %91 = sext i32 %81 to i64, !dbg !40
  %92 = getelementptr float, ptr addrspace(1) %1, i64 %91, !dbg !40
  %93 = sext i32 %82 to i64, !dbg !40
  %94 = getelementptr float, ptr addrspace(1) %1, i64 %93, !dbg !40
  %95 = sext i32 %83 to i64, !dbg !40
  %96 = getelementptr float, ptr addrspace(1) %1, i64 %95, !dbg !40
  %97 = sext i32 %84 to i64, !dbg !40
  %98 = getelementptr float, ptr addrspace(1) %1, i64 %97, !dbg !40
  %99 = sext i32 %85 to i64, !dbg !40
  %100 = getelementptr float, ptr addrspace(1) %1, i64 %99, !dbg !40
  %101 = sext i32 %86 to i64, !dbg !40
  %102 = getelementptr float, ptr addrspace(1) %1, i64 %101, !dbg !40
  %103 = add i32 %5, 15, !dbg !41
  %104 = sdiv i32 %103, 16, !dbg !43
  %105 = shl i32 %7, 4, !dbg !44
  %106 = icmp sgt i32 %103, 15, !dbg !45
  %107 = icmp slt i32 %45, %5, !dbg !46
  %108 = and i1 %107, %106, !dbg !45
  %109 = shl nuw nsw i32 %28, 4, !dbg !47
  %110 = shl nuw nsw i32 %27, 1, !dbg !47
  %111 = xor i32 %110, %25, !dbg !47
  %112 = and i32 %111, 12, !dbg !47
  %113 = and i32 %25, 3, !dbg !47
  %114 = or disjoint i32 %112, %113, !dbg !47
  %115 = or disjoint i32 %114, %109, !dbg !47
  %116 = zext nneg i32 %115 to i64, !dbg !47
  %117 = getelementptr float, ptr addrspace(3) @global_smem, i64 %116, !dbg !47
  %118 = shl nuw nsw i32 %29, 4, !dbg !47
  %119 = or disjoint i32 %114, %118, !dbg !47
  %120 = zext nneg i32 %119 to i64, !dbg !47
  %121 = getelementptr float, ptr addrspace(3) @global_smem, i64 %120, !dbg !47
  %122 = shl nuw nsw i32 %30, 4, !dbg !47
  %123 = or disjoint i32 %114, %122, !dbg !47
  %124 = zext nneg i32 %123 to i64, !dbg !47
  %125 = getelementptr float, ptr addrspace(3) @global_smem, i64 %124, !dbg !47
  %126 = shl nuw nsw i32 %31, 4, !dbg !47
  %127 = or disjoint i32 %114, %126, !dbg !47
  %128 = zext nneg i32 %127 to i64, !dbg !47
  %129 = getelementptr float, ptr addrspace(3) @global_smem, i64 %128, !dbg !47
  %130 = select i1 %108, i32 4, i32 0, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %117, ptr addrspace(1) %71, i32 %130, i1 true) #2, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %121, ptr addrspace(1) %73, i32 %130, i1 true) #2, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %125, ptr addrspace(1) %75, i32 %130, i1 true) #2, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %129, ptr addrspace(1) %77, i32 %130, i1 true) #2, !dbg !47
  tail call void asm sideeffect "cp.async.commit_group ;", ""() #2, !dbg !47
  %131 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %116, !dbg !48
  %132 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %120, !dbg !48
  %133 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %124, !dbg !48
  %134 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %128, !dbg !48
  %135 = shl nuw nsw i32 %41, 4, !dbg !48
  %136 = or disjoint i32 %114, %135, !dbg !48
  %137 = zext nneg i32 %136 to i64, !dbg !48
  %138 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %137, !dbg !48
  %139 = shl nuw nsw i32 %42, 4, !dbg !48
  %140 = or disjoint i32 %114, %139, !dbg !48
  %141 = zext nneg i32 %140 to i64, !dbg !48
  %142 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %141, !dbg !48
  %143 = shl nuw nsw i32 %43, 4, !dbg !48
  %144 = or disjoint i32 %114, %143, !dbg !48
  %145 = zext nneg i32 %144 to i64, !dbg !48
  %146 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %145, !dbg !48
  %147 = shl nuw nsw i32 %44, 4, !dbg !48
  %148 = or disjoint i32 %114, %147, !dbg !48
  %149 = zext nneg i32 %148 to i64, !dbg !48
  %150 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %149, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %131, ptr addrspace(1) %88, i32 %130, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %132, ptr addrspace(1) %90, i32 %130, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %133, ptr addrspace(1) %92, i32 %130, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %134, ptr addrspace(1) %94, i32 %130, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %138, ptr addrspace(1) %96, i32 %130, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %142, ptr addrspace(1) %98, i32 %130, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %146, ptr addrspace(1) %100, i32 %130, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %150, ptr addrspace(1) %102, i32 %130, i1 true) #2, !dbg !48
  tail call void asm sideeffect "cp.async.commit_group ;", ""() #2, !dbg !48
  %151 = icmp sgt i32 %103, 31, !dbg !45
  %152 = getelementptr i8, ptr addrspace(1) %71, i64 64, !dbg !49
  %153 = getelementptr i8, ptr addrspace(1) %73, i64 64, !dbg !49
  %154 = getelementptr i8, ptr addrspace(1) %75, i64 64, !dbg !49
  %155 = getelementptr i8, ptr addrspace(1) %77, i64 64, !dbg !49
  %156 = sext i32 %105 to i64, !dbg !50
  %157 = getelementptr float, ptr addrspace(1) %88, i64 %156, !dbg !50
  %158 = getelementptr float, ptr addrspace(1) %90, i64 %156, !dbg !50
  %159 = getelementptr float, ptr addrspace(1) %92, i64 %156, !dbg !50
  %160 = getelementptr float, ptr addrspace(1) %94, i64 %156, !dbg !50
  %161 = getelementptr float, ptr addrspace(1) %96, i64 %156, !dbg !50
  %162 = getelementptr float, ptr addrspace(1) %98, i64 %156, !dbg !50
  %163 = getelementptr float, ptr addrspace(1) %100, i64 %156, !dbg !50
  %164 = getelementptr float, ptr addrspace(1) %102, i64 %156, !dbg !50
  %165 = add i32 %5, -16, !dbg !51
  %166 = icmp slt i32 %45, %165, !dbg !46
  %167 = and i1 %151, %166, !dbg !45
  tail call void @llvm.nvvm.barrier0(), !dbg !47
  %168 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %116, !dbg !47
  %169 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %120, !dbg !47
  %170 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %124, !dbg !47
  %171 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %128, !dbg !47
  %172 = select i1 %167, i32 4, i32 0, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %168, ptr addrspace(1) %152, i32 %172, i1 true) #2, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %169, ptr addrspace(1) %153, i32 %172, i1 true) #2, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %170, ptr addrspace(1) %154, i32 %172, i1 true) #2, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %171, ptr addrspace(1) %155, i32 %172, i1 true) #2, !dbg !47
  tail call void asm sideeffect "cp.async.commit_group ;", ""() #2, !dbg !47
  %173 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 8192), i64 %116, !dbg !48
  %174 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 8192), i64 %120, !dbg !48
  %175 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 8192), i64 %124, !dbg !48
  %176 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 8192), i64 %128, !dbg !48
  %177 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 8192), i64 %137, !dbg !48
  %178 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 8192), i64 %141, !dbg !48
  %179 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 8192), i64 %145, !dbg !48
  %180 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 8192), i64 %149, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %173, ptr addrspace(1) %157, i32 %172, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %174, ptr addrspace(1) %158, i32 %172, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %175, ptr addrspace(1) %159, i32 %172, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %176, ptr addrspace(1) %160, i32 %172, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %177, ptr addrspace(1) %161, i32 %172, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %178, ptr addrspace(1) %162, i32 %172, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %179, ptr addrspace(1) %163, i32 %172, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %180, ptr addrspace(1) %164, i32 %172, i1 true) #2, !dbg !48
  tail call void asm sideeffect "cp.async.commit_group ;", ""() #2, !dbg !48
  tail call void asm sideeffect "cp.async.wait_group 0x2;", ""() #2, !dbg !47
  tail call void @llvm.nvvm.barrier0(), !dbg !47
  %181 = lshr i32 %25, 6, !dbg !47
  %182 = and i32 %181, 1, !dbg !47
  %183 = and i32 %25, 7, !dbg !47
  %184 = lshr i32 %25, 3, !dbg !47
  %185 = and i32 %184, 1, !dbg !47
  %186 = lshr i32 %26, 4, !dbg !47
  %187 = lshr i32 %183, 1, !dbg !47
  %188 = shl nuw nsw i32 %182, 4, !dbg !47
  %189 = shl nuw nsw i32 %185, 3, !dbg !47
  %190 = or disjoint i32 %188, %189, !dbg !47
  %191 = or disjoint i32 %190, %183, !dbg !47
  %192 = xor i32 %186, %187, !dbg !47
  %193 = shl nuw nsw i32 %192, 2, !dbg !47
  %194 = shl nuw nsw i32 %191, 4, !dbg !47
  %195 = or disjoint i32 %194, %193, !dbg !47
  %196 = zext nneg i32 %195 to i64, !dbg !47
  %197 = getelementptr float, ptr addrspace(3) @global_smem, i64 %196, !dbg !47
  %198 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,r"(ptr addrspace(3) %197) #2, !dbg !47
  %199 = lshr i32 %25, 2, !dbg !48
  %200 = and i32 %199, 8, !dbg !48
  %201 = and i32 %25, 23, !dbg !48
  %202 = or disjoint i32 %201, %200, !dbg !48
  %203 = xor i32 %185, %187, !dbg !48
  %204 = shl nuw nsw i32 %203, 2, !dbg !48
  %205 = shl nuw nsw i32 %202, 4, !dbg !48
  %206 = or disjoint i32 %205, %204, !dbg !48
  %207 = zext nneg i32 %206 to i64, !dbg !48
  %208 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %207, !dbg !48
  %209 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,r"(ptr addrspace(3) %208) #2, !dbg !48
  %210 = getelementptr i8, ptr addrspace(3) %208, i64 2048, !dbg !48
  %211 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,r"(ptr addrspace(3) %210) #2, !dbg !48
  br i1 %106, label %.lr.ph, label %._crit_edge, !dbg !45

.lr.ph:                                           ; preds = %9
  %212 = add nsw i32 %104, -2
  %213 = or disjoint i32 %186, 2
  %214 = xor i32 %213, %187
  %215 = shl nuw nsw i32 %214, 2
  %216 = or disjoint i32 %185, 2
  %217 = xor i32 %216, %187
  %218 = shl nuw nsw i32 %217, 2
  %.neg62 = add nsw i32 %5, -32
  %219 = shl nuw nsw i32 %191, 4
  %220 = or disjoint i32 %219, %215
  %221 = zext nneg i32 %220 to i64
  %222 = shl nuw nsw i32 %202, 4
  %223 = or disjoint i32 %222, %218
  %224 = zext nneg i32 %223 to i64
  br label %225, !dbg !45

225:                                              ; preds = %.lr.ph, %225
  %.pn = phi { i32, i32, i32, i32 } [ %211, %.lr.ph ], [ %350, %225 ]
  %.pn83 = phi { i32, i32, i32, i32 } [ %209, %.lr.ph ], [ %348, %225 ]
  %.pn87 = phi { i32, i32, i32, i32 } [ %198, %.lr.ph ], [ %346, %225 ]
  %226 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), %.lr.ph ], [ %344, %225 ]
  %227 = phi ptr addrspace(3) [ @global_smem, %.lr.ph ], [ %341, %225 ]
  %228 = phi i32 [ 0, %.lr.ph ], [ %338, %225 ]
  %229 = phi i32 [ 1, %.lr.ph ], [ %312, %225 ]
  %.pn1174 = phi ptr addrspace(1) [ %164, %.lr.ph ], [ %309, %225 ]
  %.pn1373 = phi ptr addrspace(1) [ %163, %.lr.ph ], [ %308, %225 ]
  %.pn1572 = phi ptr addrspace(1) [ %162, %.lr.ph ], [ %307, %225 ]
  %.pn1771 = phi ptr addrspace(1) [ %161, %.lr.ph ], [ %306, %225 ]
  %.pn1970 = phi ptr addrspace(1) [ %160, %.lr.ph ], [ %305, %225 ]
  %.pn2169 = phi ptr addrspace(1) [ %159, %.lr.ph ], [ %304, %225 ]
  %.pn2368 = phi ptr addrspace(1) [ %158, %.lr.ph ], [ %303, %225 ]
  %.pn2567 = phi ptr addrspace(1) [ %157, %.lr.ph ], [ %302, %225 ]
  %.pn366 = phi ptr addrspace(1) [ %155, %.lr.ph ], [ %301, %225 ]
  %.pn565 = phi ptr addrspace(1) [ %154, %.lr.ph ], [ %300, %225 ]
  %.pn764 = phi ptr addrspace(1) [ %153, %.lr.ph ], [ %299, %225 ]
  %.pn963 = phi ptr addrspace(1) [ %152, %.lr.ph ], [ %298, %225 ]
  %230 = phi float [ 0.000000e+00, %.lr.ph ], [ %352, %225 ]
  %231 = phi float [ 0.000000e+00, %.lr.ph ], [ %353, %225 ]
  %232 = phi float [ 0.000000e+00, %.lr.ph ], [ %354, %225 ]
  %233 = phi float [ 0.000000e+00, %.lr.ph ], [ %355, %225 ]
  %234 = phi float [ 0.000000e+00, %.lr.ph ], [ %357, %225 ]
  %235 = phi float [ 0.000000e+00, %.lr.ph ], [ %358, %225 ]
  %236 = phi float [ 0.000000e+00, %.lr.ph ], [ %359, %225 ]
  %237 = phi float [ 0.000000e+00, %.lr.ph ], [ %360, %225 ]
  %238 = phi float [ 0.000000e+00, %.lr.ph ], [ %362, %225 ]
  %239 = phi float [ 0.000000e+00, %.lr.ph ], [ %363, %225 ]
  %240 = phi float [ 0.000000e+00, %.lr.ph ], [ %364, %225 ]
  %241 = phi float [ 0.000000e+00, %.lr.ph ], [ %365, %225 ]
  %242 = phi float [ 0.000000e+00, %.lr.ph ], [ %367, %225 ]
  %243 = phi float [ 0.000000e+00, %.lr.ph ], [ %368, %225 ]
  %244 = phi float [ 0.000000e+00, %.lr.ph ], [ %369, %225 ]
  %245 = phi float [ 0.000000e+00, %.lr.ph ], [ %370, %225 ]
  %246 = phi i32 [ 0, %.lr.ph ], [ %371, %225 ]
  %247 = extractvalue { i32, i32, i32, i32 } %.pn87, 3, !dbg !45
  %248 = extractvalue { i32, i32, i32, i32 } %.pn87, 2, !dbg !45
  %249 = extractvalue { i32, i32, i32, i32 } %.pn87, 1, !dbg !45
  %250 = extractvalue { i32, i32, i32, i32 } %.pn87, 0, !dbg !45
  %251 = extractvalue { i32, i32, i32, i32 } %.pn83, 3, !dbg !45
  %252 = extractvalue { i32, i32, i32, i32 } %.pn83, 2, !dbg !45
  %253 = extractvalue { i32, i32, i32, i32 } %.pn83, 1, !dbg !45
  %254 = extractvalue { i32, i32, i32, i32 } %.pn83, 0, !dbg !45
  %255 = extractvalue { i32, i32, i32, i32 } %.pn, 3, !dbg !45
  %256 = extractvalue { i32, i32, i32, i32 } %.pn, 2, !dbg !45
  %257 = extractvalue { i32, i32, i32, i32 } %.pn, 1, !dbg !45
  %258 = extractvalue { i32, i32, i32, i32 } %.pn, 0, !dbg !45
  %259 = icmp slt i32 %246, %212, !dbg !45
  %260 = getelementptr float, ptr addrspace(3) %227, i64 %221, !dbg !47
  %261 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,r"(ptr addrspace(3) %260) #2, !dbg !47
  %262 = extractvalue { i32, i32, i32, i32 } %261, 0, !dbg !47
  %263 = extractvalue { i32, i32, i32, i32 } %261, 1, !dbg !47
  %264 = extractvalue { i32, i32, i32, i32 } %261, 2, !dbg !47
  %265 = extractvalue { i32, i32, i32, i32 } %261, 3, !dbg !47
  %266 = getelementptr float, ptr addrspace(3) %226, i64 %224, !dbg !48
  %267 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,r"(ptr addrspace(3) %266) #2, !dbg !48
  %268 = extractvalue { i32, i32, i32, i32 } %267, 0, !dbg !48
  %269 = extractvalue { i32, i32, i32, i32 } %267, 1, !dbg !48
  %270 = extractvalue { i32, i32, i32, i32 } %267, 2, !dbg !48
  %271 = extractvalue { i32, i32, i32, i32 } %267, 3, !dbg !48
  %272 = getelementptr i8, ptr addrspace(3) %266, i64 2048, !dbg !48
  %273 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,r"(ptr addrspace(3) %272) #2, !dbg !48
  %274 = extractvalue { i32, i32, i32, i32 } %273, 0, !dbg !48
  %275 = extractvalue { i32, i32, i32, i32 } %273, 1, !dbg !48
  %276 = extractvalue { i32, i32, i32, i32 } %273, 2, !dbg !48
  %277 = extractvalue { i32, i32, i32, i32 } %273, 3, !dbg !48
  %278 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %230, float %231, float %232, float %233, i32 %250, i32 %249, i32 %248, i32 %247, i32 %254, i32 %253) #2, !dbg !52
  %279 = extractvalue { float, float, float, float } %278, 0, !dbg !52
  %280 = extractvalue { float, float, float, float } %278, 1, !dbg !52
  %281 = extractvalue { float, float, float, float } %278, 2, !dbg !52
  %282 = extractvalue { float, float, float, float } %278, 3, !dbg !52
  %283 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %234, float %235, float %236, float %237, i32 %250, i32 %249, i32 %248, i32 %247, i32 %252, i32 %251) #2, !dbg !52
  %284 = extractvalue { float, float, float, float } %283, 0, !dbg !52
  %285 = extractvalue { float, float, float, float } %283, 1, !dbg !52
  %286 = extractvalue { float, float, float, float } %283, 2, !dbg !52
  %287 = extractvalue { float, float, float, float } %283, 3, !dbg !52
  %288 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %238, float %239, float %240, float %241, i32 %250, i32 %249, i32 %248, i32 %247, i32 %258, i32 %257) #2, !dbg !52
  %289 = extractvalue { float, float, float, float } %288, 0, !dbg !52
  %290 = extractvalue { float, float, float, float } %288, 1, !dbg !52
  %291 = extractvalue { float, float, float, float } %288, 2, !dbg !52
  %292 = extractvalue { float, float, float, float } %288, 3, !dbg !52
  %293 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %242, float %243, float %244, float %245, i32 %250, i32 %249, i32 %248, i32 %247, i32 %256, i32 %255) #2, !dbg !52
  %294 = extractvalue { float, float, float, float } %293, 0, !dbg !52
  %295 = extractvalue { float, float, float, float } %293, 1, !dbg !52
  %296 = extractvalue { float, float, float, float } %293, 2, !dbg !52
  %297 = extractvalue { float, float, float, float } %293, 3, !dbg !52
  %298 = getelementptr i8, ptr addrspace(1) %.pn963, i64 64, !dbg !49
  %299 = getelementptr i8, ptr addrspace(1) %.pn764, i64 64, !dbg !49
  %300 = getelementptr i8, ptr addrspace(1) %.pn565, i64 64, !dbg !49
  %301 = getelementptr i8, ptr addrspace(1) %.pn366, i64 64, !dbg !49
  %302 = getelementptr float, ptr addrspace(1) %.pn2567, i64 %156, !dbg !50
  %303 = getelementptr float, ptr addrspace(1) %.pn2368, i64 %156, !dbg !50
  %304 = getelementptr float, ptr addrspace(1) %.pn2169, i64 %156, !dbg !50
  %305 = getelementptr float, ptr addrspace(1) %.pn1970, i64 %156, !dbg !50
  %306 = getelementptr float, ptr addrspace(1) %.pn1771, i64 %156, !dbg !50
  %307 = getelementptr float, ptr addrspace(1) %.pn1572, i64 %156, !dbg !50
  %308 = getelementptr float, ptr addrspace(1) %.pn1373, i64 %156, !dbg !50
  %309 = getelementptr float, ptr addrspace(1) %.pn1174, i64 %156, !dbg !50
  %310 = add i32 %229, 1, !dbg !45
  %311 = icmp slt i32 %310, 2, !dbg !45
  %312 = select i1 %311, i32 %310, i32 0, !dbg !45
  %313 = shl i32 %246, 4, !dbg !51
  %314 = sub i32 %.neg62, %313, !dbg !51
  %315 = icmp slt i32 %45, %314, !dbg !46
  %316 = shl i32 %312, 9, !dbg !47
  %317 = sext i32 %316 to i64, !dbg !47
  %318 = getelementptr float, ptr addrspace(3) @global_smem, i64 %317, !dbg !47
  %319 = and i1 %259, %315, !dbg !45
  tail call void @llvm.nvvm.barrier0(), !dbg !47
  %320 = getelementptr float, ptr addrspace(3) %318, i64 %116, !dbg !47
  %321 = getelementptr float, ptr addrspace(3) %318, i64 %120, !dbg !47
  %322 = getelementptr float, ptr addrspace(3) %318, i64 %124, !dbg !47
  %323 = getelementptr float, ptr addrspace(3) %318, i64 %128, !dbg !47
  %324 = select i1 %319, i32 4, i32 0, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %320, ptr addrspace(1) %298, i32 %324, i1 true) #2, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %321, ptr addrspace(1) %299, i32 %324, i1 true) #2, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %322, ptr addrspace(1) %300, i32 %324, i1 true) #2, !dbg !47
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %323, ptr addrspace(1) %301, i32 %324, i1 true) #2, !dbg !47
  tail call void asm sideeffect "cp.async.commit_group ;", ""() #2, !dbg !47
  %325 = shl i32 %312, 10, !dbg !48
  %326 = sext i32 %325 to i64, !dbg !48
  %327 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %326, !dbg !48
  %328 = getelementptr float, ptr addrspace(3) %327, i64 %116, !dbg !48
  %329 = getelementptr float, ptr addrspace(3) %327, i64 %120, !dbg !48
  %330 = getelementptr float, ptr addrspace(3) %327, i64 %124, !dbg !48
  %331 = getelementptr float, ptr addrspace(3) %327, i64 %128, !dbg !48
  %332 = getelementptr float, ptr addrspace(3) %327, i64 %137, !dbg !48
  %333 = getelementptr float, ptr addrspace(3) %327, i64 %141, !dbg !48
  %334 = getelementptr float, ptr addrspace(3) %327, i64 %145, !dbg !48
  %335 = getelementptr float, ptr addrspace(3) %327, i64 %149, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %328, ptr addrspace(1) %302, i32 %324, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %329, ptr addrspace(1) %303, i32 %324, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %330, ptr addrspace(1) %304, i32 %324, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %331, ptr addrspace(1) %305, i32 %324, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %332, ptr addrspace(1) %306, i32 %324, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %333, ptr addrspace(1) %307, i32 %324, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %334, ptr addrspace(1) %308, i32 %324, i1 true) #2, !dbg !48
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %335, ptr addrspace(1) %309, i32 %324, i1 true) #2, !dbg !48
  tail call void asm sideeffect "cp.async.commit_group ;", ""() #2, !dbg !48
  %336 = add i32 %228, 1, !dbg !45
  %337 = icmp slt i32 %336, 2, !dbg !45
  %338 = select i1 %337, i32 %336, i32 0, !dbg !45
  %339 = shl i32 %338, 9, !dbg !47
  %340 = sext i32 %339 to i64, !dbg !47
  %341 = getelementptr float, ptr addrspace(3) @global_smem, i64 %340, !dbg !47
  tail call void asm sideeffect "cp.async.wait_group 0x2;", ""() #2, !dbg !47
  tail call void @llvm.nvvm.barrier0(), !dbg !47
  %342 = shl i32 %338, 10, !dbg !48
  %343 = sext i32 %342 to i64, !dbg !48
  %344 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 4096), i64 %343, !dbg !48
  %345 = getelementptr float, ptr addrspace(3) %341, i64 %196, !dbg !47
  %346 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,r"(ptr addrspace(3) %345) #2, !dbg !47
  %347 = getelementptr float, ptr addrspace(3) %344, i64 %207, !dbg !48
  %348 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,r"(ptr addrspace(3) %347) #2, !dbg !48
  %349 = getelementptr i8, ptr addrspace(3) %347, i64 2048, !dbg !48
  %350 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,r"(ptr addrspace(3) %349) #2, !dbg !48
  %351 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %279, float %280, float %281, float %282, i32 %262, i32 %263, i32 %264, i32 %265, i32 %268, i32 %269) #2, !dbg !52
  %352 = extractvalue { float, float, float, float } %351, 0, !dbg !52
  %353 = extractvalue { float, float, float, float } %351, 1, !dbg !52
  %354 = extractvalue { float, float, float, float } %351, 2, !dbg !52
  %355 = extractvalue { float, float, float, float } %351, 3, !dbg !52
  %356 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %284, float %285, float %286, float %287, i32 %262, i32 %263, i32 %264, i32 %265, i32 %270, i32 %271) #2, !dbg !52
  %357 = extractvalue { float, float, float, float } %356, 0, !dbg !52
  %358 = extractvalue { float, float, float, float } %356, 1, !dbg !52
  %359 = extractvalue { float, float, float, float } %356, 2, !dbg !52
  %360 = extractvalue { float, float, float, float } %356, 3, !dbg !52
  %361 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %289, float %290, float %291, float %292, i32 %262, i32 %263, i32 %264, i32 %265, i32 %274, i32 %275) #2, !dbg !52
  %362 = extractvalue { float, float, float, float } %361, 0, !dbg !52
  %363 = extractvalue { float, float, float, float } %361, 1, !dbg !52
  %364 = extractvalue { float, float, float, float } %361, 2, !dbg !52
  %365 = extractvalue { float, float, float, float } %361, 3, !dbg !52
  %366 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %294, float %295, float %296, float %297, i32 %262, i32 %263, i32 %264, i32 %265, i32 %276, i32 %277) #2, !dbg !52
  %367 = extractvalue { float, float, float, float } %366, 0, !dbg !52
  %368 = extractvalue { float, float, float, float } %366, 1, !dbg !52
  %369 = extractvalue { float, float, float, float } %366, 2, !dbg !52
  %370 = extractvalue { float, float, float, float } %366, 3, !dbg !52
  %371 = add nuw nsw i32 %246, 1, !dbg !45
  %372 = icmp slt i32 %371, %104, !dbg !45
  br i1 %372, label %225, label %._crit_edge, !dbg !45

._crit_edge:                                      ; preds = %225, %9
  %373 = phi float [ 0.000000e+00, %9 ], [ %352, %225 ]
  %374 = phi float [ 0.000000e+00, %9 ], [ %353, %225 ]
  %375 = phi float [ 0.000000e+00, %9 ], [ %354, %225 ]
  %376 = phi float [ 0.000000e+00, %9 ], [ %355, %225 ]
  %377 = phi float [ 0.000000e+00, %9 ], [ %357, %225 ]
  %378 = phi float [ 0.000000e+00, %9 ], [ %358, %225 ]
  %379 = phi float [ 0.000000e+00, %9 ], [ %359, %225 ]
  %380 = phi float [ 0.000000e+00, %9 ], [ %360, %225 ]
  %381 = phi float [ 0.000000e+00, %9 ], [ %362, %225 ]
  %382 = phi float [ 0.000000e+00, %9 ], [ %363, %225 ]
  %383 = phi float [ 0.000000e+00, %9 ], [ %364, %225 ]
  %384 = phi float [ 0.000000e+00, %9 ], [ %365, %225 ]
  %385 = phi float [ 0.000000e+00, %9 ], [ %367, %225 ]
  %386 = phi float [ 0.000000e+00, %9 ], [ %368, %225 ]
  %387 = phi float [ 0.000000e+00, %9 ], [ %369, %225 ]
  %388 = phi float [ 0.000000e+00, %9 ], [ %370, %225 ]
  %389 = and i32 %25, 63, !dbg !32
  %390 = or disjoint i32 %40, %389, !dbg !33
  %391 = or disjoint i32 %182, %24, !dbg !29
  %392 = or disjoint i32 %391, 30, !dbg !29
  %393 = or disjoint i32 %391, 28, !dbg !29
  %394 = or disjoint i32 %391, 26, !dbg !29
  %395 = or disjoint i32 %391, 24, !dbg !29
  %396 = or disjoint i32 %391, 22, !dbg !29
  %397 = or disjoint i32 %391, 20, !dbg !29
  %398 = or disjoint i32 %391, 18, !dbg !29
  %399 = or disjoint i32 %391, 16, !dbg !29
  %400 = or disjoint i32 %391, 14, !dbg !29
  %401 = or disjoint i32 %391, 12, !dbg !29
  %402 = or disjoint i32 %391, 10, !dbg !29
  %403 = or disjoint i32 %391, 8, !dbg !29
  %404 = or disjoint i32 %391, 6, !dbg !29
  %405 = or disjoint i32 %391, 4, !dbg !29
  %406 = or disjoint i32 %391, 2, !dbg !29
  tail call void asm sideeffect "cp.async.wait_group 0x0;", ""() #2, !dbg !45
  tail call void @llvm.nvvm.barrier0(), !dbg !45
  %407 = mul i32 %391, %8, !dbg !53
  %408 = mul i32 %406, %8, !dbg !53
  %409 = mul i32 %405, %8, !dbg !53
  %410 = mul i32 %404, %8, !dbg !53
  %411 = mul i32 %403, %8, !dbg !53
  %412 = mul i32 %402, %8, !dbg !53
  %413 = mul i32 %401, %8, !dbg !53
  %414 = mul i32 %400, %8, !dbg !53
  %415 = mul i32 %399, %8, !dbg !53
  %416 = mul i32 %398, %8, !dbg !53
  %417 = mul i32 %397, %8, !dbg !53
  %418 = mul i32 %396, %8, !dbg !53
  %419 = mul i32 %395, %8, !dbg !53
  %420 = mul i32 %394, %8, !dbg !53
  %421 = mul i32 %393, %8, !dbg !53
  %422 = mul i32 %392, %8, !dbg !53
  %423 = sext i32 %407 to i64, !dbg !54
  %424 = getelementptr float, ptr addrspace(1) %2, i64 %423, !dbg !54
  %425 = sext i32 %408 to i64, !dbg !54
  %426 = getelementptr float, ptr addrspace(1) %2, i64 %425, !dbg !54
  %427 = sext i32 %409 to i64, !dbg !54
  %428 = getelementptr float, ptr addrspace(1) %2, i64 %427, !dbg !54
  %429 = sext i32 %410 to i64, !dbg !54
  %430 = getelementptr float, ptr addrspace(1) %2, i64 %429, !dbg !54
  %431 = sext i32 %411 to i64, !dbg !54
  %432 = getelementptr float, ptr addrspace(1) %2, i64 %431, !dbg !54
  %433 = sext i32 %412 to i64, !dbg !54
  %434 = getelementptr float, ptr addrspace(1) %2, i64 %433, !dbg !54
  %435 = sext i32 %413 to i64, !dbg !54
  %436 = getelementptr float, ptr addrspace(1) %2, i64 %435, !dbg !54
  %437 = sext i32 %414 to i64, !dbg !54
  %438 = getelementptr float, ptr addrspace(1) %2, i64 %437, !dbg !54
  %439 = sext i32 %415 to i64, !dbg !54
  %440 = getelementptr float, ptr addrspace(1) %2, i64 %439, !dbg !54
  %441 = sext i32 %416 to i64, !dbg !54
  %442 = getelementptr float, ptr addrspace(1) %2, i64 %441, !dbg !54
  %443 = sext i32 %417 to i64, !dbg !54
  %444 = getelementptr float, ptr addrspace(1) %2, i64 %443, !dbg !54
  %445 = sext i32 %418 to i64, !dbg !54
  %446 = getelementptr float, ptr addrspace(1) %2, i64 %445, !dbg !54
  %447 = sext i32 %419 to i64, !dbg !54
  %448 = getelementptr float, ptr addrspace(1) %2, i64 %447, !dbg !54
  %449 = sext i32 %420 to i64, !dbg !54
  %450 = getelementptr float, ptr addrspace(1) %2, i64 %449, !dbg !54
  %451 = sext i32 %421 to i64, !dbg !54
  %452 = getelementptr float, ptr addrspace(1) %2, i64 %451, !dbg !54
  %453 = sext i32 %422 to i64, !dbg !54
  %454 = getelementptr float, ptr addrspace(1) %2, i64 %453, !dbg !54
  %455 = sext i32 %390 to i64, !dbg !55
  %456 = getelementptr float, ptr addrspace(1) %424, i64 %455, !dbg !55
  %457 = getelementptr float, ptr addrspace(1) %426, i64 %455, !dbg !55
  %458 = getelementptr float, ptr addrspace(1) %428, i64 %455, !dbg !55
  %459 = getelementptr float, ptr addrspace(1) %430, i64 %455, !dbg !55
  %460 = getelementptr float, ptr addrspace(1) %432, i64 %455, !dbg !55
  %461 = getelementptr float, ptr addrspace(1) %434, i64 %455, !dbg !55
  %462 = getelementptr float, ptr addrspace(1) %436, i64 %455, !dbg !55
  %463 = getelementptr float, ptr addrspace(1) %438, i64 %455, !dbg !55
  %464 = getelementptr float, ptr addrspace(1) %440, i64 %455, !dbg !55
  %465 = getelementptr float, ptr addrspace(1) %442, i64 %455, !dbg !55
  %466 = getelementptr float, ptr addrspace(1) %444, i64 %455, !dbg !55
  %467 = getelementptr float, ptr addrspace(1) %446, i64 %455, !dbg !55
  %468 = getelementptr float, ptr addrspace(1) %448, i64 %455, !dbg !55
  %469 = getelementptr float, ptr addrspace(1) %450, i64 %455, !dbg !55
  %470 = getelementptr float, ptr addrspace(1) %452, i64 %455, !dbg !55
  %471 = getelementptr float, ptr addrspace(1) %454, i64 %455, !dbg !55
  %472 = icmp slt i32 %391, %3, !dbg !56
  %473 = icmp slt i32 %406, %3, !dbg !56
  %474 = icmp slt i32 %405, %3, !dbg !56
  %475 = icmp slt i32 %404, %3, !dbg !56
  %476 = icmp slt i32 %403, %3, !dbg !56
  %477 = icmp slt i32 %402, %3, !dbg !56
  %478 = icmp slt i32 %401, %3, !dbg !56
  %479 = icmp slt i32 %400, %3, !dbg !56
  %480 = icmp slt i32 %399, %3, !dbg !56
  %481 = icmp slt i32 %398, %3, !dbg !56
  %482 = icmp slt i32 %397, %3, !dbg !56
  %483 = icmp slt i32 %396, %3, !dbg !56
  %484 = icmp slt i32 %395, %3, !dbg !56
  %485 = icmp slt i32 %394, %3, !dbg !56
  %486 = icmp slt i32 %393, %3, !dbg !56
  %487 = icmp slt i32 %392, %3, !dbg !56
  %488 = icmp slt i32 %390, %4, !dbg !57
  %489 = and i1 %472, %488, !dbg !58
  %490 = and i1 %473, %488, !dbg !58
  %491 = and i1 %474, %488, !dbg !58
  %492 = and i1 %475, %488, !dbg !58
  %493 = and i1 %476, %488, !dbg !58
  %494 = and i1 %477, %488, !dbg !58
  %495 = and i1 %478, %488, !dbg !58
  %496 = and i1 %479, %488, !dbg !58
  %497 = and i1 %480, %488, !dbg !58
  %498 = and i1 %481, %488, !dbg !58
  %499 = and i1 %482, %488, !dbg !58
  %500 = and i1 %483, %488, !dbg !58
  %501 = and i1 %484, %488, !dbg !58
  %502 = and i1 %485, %488, !dbg !58
  %503 = and i1 %486, %488, !dbg !58
  %504 = and i1 %487, %488, !dbg !58
  %505 = lshr i32 %26, 2, !dbg !59
  %506 = shl i32 %25, 1, !dbg !59
  %507 = and i32 %506, 6, !dbg !59
  %508 = or disjoint i32 %188, %505, !dbg !59
  %509 = or disjoint i32 %507, %200, !dbg !59
  %510 = mul nuw nsw i32 %508, 66, !dbg !59
  %511 = add nuw nsw i32 %510, %509, !dbg !59
  %512 = zext nneg i32 %511 to i64, !dbg !59
  %513 = getelementptr float, ptr addrspace(3) @global_smem, i64 %512, !dbg !59
  %514 = insertelement <2 x float> poison, float %373, i64 0, !dbg !59
  %515 = insertelement <2 x float> %514, float %374, i64 1, !dbg !59
  store <2 x float> %515, ptr addrspace(3) %513, align 8, !dbg !59
  %516 = add nuw nsw i32 %510, 528, !dbg !59
  %517 = add nuw nsw i32 %516, %509, !dbg !59
  %518 = zext nneg i32 %517 to i64, !dbg !59
  %519 = getelementptr float, ptr addrspace(3) @global_smem, i64 %518, !dbg !59
  %520 = insertelement <2 x float> poison, float %375, i64 0, !dbg !59
  %521 = insertelement <2 x float> %520, float %376, i64 1, !dbg !59
  store <2 x float> %521, ptr addrspace(3) %519, align 8, !dbg !59
  %522 = or disjoint i32 %509, 16, !dbg !59
  %523 = add nuw nsw i32 %510, %522, !dbg !59
  %524 = zext nneg i32 %523 to i64, !dbg !59
  %525 = getelementptr float, ptr addrspace(3) @global_smem, i64 %524, !dbg !59
  %526 = insertelement <2 x float> poison, float %377, i64 0, !dbg !59
  %527 = insertelement <2 x float> %526, float %378, i64 1, !dbg !59
  store <2 x float> %527, ptr addrspace(3) %525, align 8, !dbg !59
  %528 = add nuw nsw i32 %516, %522, !dbg !59
  %529 = zext nneg i32 %528 to i64, !dbg !59
  %530 = getelementptr float, ptr addrspace(3) @global_smem, i64 %529, !dbg !59
  %531 = insertelement <2 x float> poison, float %379, i64 0, !dbg !59
  %532 = insertelement <2 x float> %531, float %380, i64 1, !dbg !59
  store <2 x float> %532, ptr addrspace(3) %530, align 8, !dbg !59
  %533 = or disjoint i32 %509, 32, !dbg !59
  %534 = add nuw nsw i32 %510, %533, !dbg !59
  %535 = zext nneg i32 %534 to i64, !dbg !59
  %536 = getelementptr float, ptr addrspace(3) @global_smem, i64 %535, !dbg !59
  %537 = insertelement <2 x float> poison, float %381, i64 0, !dbg !59
  %538 = insertelement <2 x float> %537, float %382, i64 1, !dbg !59
  store <2 x float> %538, ptr addrspace(3) %536, align 8, !dbg !59
  %539 = add nuw nsw i32 %516, %533, !dbg !59
  %540 = zext nneg i32 %539 to i64, !dbg !59
  %541 = getelementptr float, ptr addrspace(3) @global_smem, i64 %540, !dbg !59
  %542 = insertelement <2 x float> poison, float %383, i64 0, !dbg !59
  %543 = insertelement <2 x float> %542, float %384, i64 1, !dbg !59
  store <2 x float> %543, ptr addrspace(3) %541, align 8, !dbg !59
  %544 = or disjoint i32 %509, 48, !dbg !59
  %545 = add nuw nsw i32 %510, %544, !dbg !59
  %546 = zext nneg i32 %545 to i64, !dbg !59
  %547 = getelementptr float, ptr addrspace(3) @global_smem, i64 %546, !dbg !59
  %548 = insertelement <2 x float> poison, float %385, i64 0, !dbg !59
  %549 = insertelement <2 x float> %548, float %386, i64 1, !dbg !59
  store <2 x float> %549, ptr addrspace(3) %547, align 8, !dbg !59
  %550 = add nuw nsw i32 %516, %544, !dbg !59
  %551 = zext nneg i32 %550 to i64, !dbg !59
  %552 = getelementptr float, ptr addrspace(3) @global_smem, i64 %551, !dbg !59
  %553 = insertelement <2 x float> poison, float %387, i64 0, !dbg !59
  %554 = insertelement <2 x float> %553, float %388, i64 1, !dbg !59
  store <2 x float> %554, ptr addrspace(3) %552, align 8, !dbg !59
  tail call void @llvm.nvvm.barrier0(), !dbg !59
  %555 = mul nuw nsw i32 %182, 66, !dbg !59
  %556 = add nuw nsw i32 %555, %389, !dbg !59
  %557 = zext nneg i32 %556 to i64, !dbg !59
  %558 = getelementptr float, ptr addrspace(3) @global_smem, i64 %557, !dbg !59
  %559 = load i32, ptr addrspace(3) %558, align 4, !dbg !59
  %560 = getelementptr i8, ptr addrspace(3) %558, i64 528, !dbg !59
  %561 = load i32, ptr addrspace(3) %560, align 4, !dbg !59
  %562 = getelementptr i8, ptr addrspace(3) %558, i64 1056, !dbg !59
  %563 = load i32, ptr addrspace(3) %562, align 4, !dbg !59
  %564 = getelementptr i8, ptr addrspace(3) %558, i64 1584, !dbg !59
  %565 = load i32, ptr addrspace(3) %564, align 4, !dbg !59
  %566 = getelementptr i8, ptr addrspace(3) %558, i64 2112, !dbg !59
  %567 = load i32, ptr addrspace(3) %566, align 4, !dbg !59
  %568 = getelementptr i8, ptr addrspace(3) %558, i64 2640, !dbg !59
  %569 = load i32, ptr addrspace(3) %568, align 4, !dbg !59
  %570 = getelementptr i8, ptr addrspace(3) %558, i64 3168, !dbg !59
  %571 = load i32, ptr addrspace(3) %570, align 4, !dbg !59
  %572 = getelementptr i8, ptr addrspace(3) %558, i64 3696, !dbg !59
  %573 = load i32, ptr addrspace(3) %572, align 4, !dbg !59
  %574 = getelementptr i8, ptr addrspace(3) %558, i64 4224, !dbg !59
  %575 = load i32, ptr addrspace(3) %574, align 4, !dbg !59
  %576 = getelementptr i8, ptr addrspace(3) %558, i64 4752, !dbg !59
  %577 = load i32, ptr addrspace(3) %576, align 4, !dbg !59
  %578 = getelementptr i8, ptr addrspace(3) %558, i64 5280, !dbg !59
  %579 = load i32, ptr addrspace(3) %578, align 4, !dbg !59
  %580 = getelementptr i8, ptr addrspace(3) %558, i64 5808, !dbg !59
  %581 = load i32, ptr addrspace(3) %580, align 4, !dbg !59
  %582 = getelementptr i8, ptr addrspace(3) %558, i64 6336, !dbg !59
  %583 = load i32, ptr addrspace(3) %582, align 4, !dbg !59
  %584 = getelementptr i8, ptr addrspace(3) %558, i64 6864, !dbg !59
  %585 = load i32, ptr addrspace(3) %584, align 4, !dbg !59
  %586 = getelementptr i8, ptr addrspace(3) %558, i64 7392, !dbg !59
  %587 = load i32, ptr addrspace(3) %586, align 4, !dbg !59
  %588 = getelementptr i8, ptr addrspace(3) %558, i64 7920, !dbg !59
  %589 = load i32, ptr addrspace(3) %588, align 4, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %559, ptr addrspace(1) %456, i1 %489) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %561, ptr addrspace(1) %457, i1 %490) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %563, ptr addrspace(1) %458, i1 %491) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %565, ptr addrspace(1) %459, i1 %492) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %567, ptr addrspace(1) %460, i1 %493) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %569, ptr addrspace(1) %461, i1 %494) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %571, ptr addrspace(1) %462, i1 %495) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %573, ptr addrspace(1) %463, i1 %496) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %575, ptr addrspace(1) %464, i1 %497) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %577, ptr addrspace(1) %465, i1 %498) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %579, ptr addrspace(1) %466, i1 %499) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %581, ptr addrspace(1) %467, i1 %500) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %583, ptr addrspace(1) %468, i1 %501) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %585, ptr addrspace(1) %469, i1 %502) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %587, ptr addrspace(1) %470, i1 %503) #2, !dbg !59
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %589, ptr addrspace(1) %471, i1 %504) #2, !dbg !59
  ret void, !dbg !60
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #1

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!nvvm.annotations = !{!4, !5}
!llvm.ident = !{!6}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!3 = !DIFile(filename: "test_matmul.py", directory: "/home/yongqi/external/triton_shared/python/examples")
!4 = !{ptr @matmul_kernel, !"kernel", i32 1}
!5 = !{ptr @matmul_kernel, !"maxntidx", i32 128}
!6 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!7 = distinct !DISubprogram(name: "matmul_kernel", linkageName: "matmul_kernel", scope: !3, file: !3, line: 37, type: !8, scopeLine: 37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!8 = !DISubroutineType(cc: DW_CC_normal, types: !9)
!9 = !{}
!10 = !DILocation(line: 60, column: 24, scope: !7)
!11 = !DILocation(line: 44, column: 22, scope: !12, inlinedAt: !14)
!12 = distinct !DILexicalBlockFile(scope: !7, file: !13, discriminator: 0)
!13 = !DIFile(filename: "standard.py", directory: "/home/yongqi/.local/lib/python3.12/site-packages/triton/language")
!14 = !DILocation(line: 61, column: 27, scope: !7)
!15 = !DILocation(line: 44, column: 28, scope: !12, inlinedAt: !14)
!16 = !DILocation(line: 44, column: 22, scope: !12, inlinedAt: !17)
!17 = !DILocation(line: 62, column: 27, scope: !7)
!18 = !DILocation(line: 44, column: 28, scope: !12, inlinedAt: !17)
!19 = !DILocation(line: 63, column: 38, scope: !7)
!20 = !DILocation(line: 64, column: 22, scope: !7)
!21 = !DILocation(line: 65, column: 29, scope: !7)
!22 = !DILocation(line: 66, column: 35, scope: !7)
!23 = !DILocation(line: 66, column: 48, scope: !7)
!24 = !DILocation(line: 67, column: 33, scope: !7)
!25 = !DILocation(line: 67, column: 27, scope: !7)
!26 = !DILocation(line: 68, column: 40, scope: !7)
!27 = !DILocation(line: 77, column: 23, scope: !7)
!28 = !DILocation(line: 77, column: 51, scope: !7)
!29 = !DILocation(line: 77, column: 38, scope: !7)
!30 = !DILocation(line: 77, column: 68, scope: !7)
!31 = !DILocation(line: 78, column: 23, scope: !7)
!32 = !DILocation(line: 78, column: 51, scope: !7)
!33 = !DILocation(line: 78, column: 38, scope: !7)
!34 = !DILocation(line: 78, column: 68, scope: !7)
!35 = !DILocation(line: 80, column: 41, scope: !7)
!36 = !DILocation(line: 80, column: 53, scope: !7)
!37 = !DILocation(line: 80, column: 22, scope: !7)
!38 = !DILocation(line: 81, column: 40, scope: !7)
!39 = !DILocation(line: 81, column: 52, scope: !7)
!40 = !DILocation(line: 81, column: 22, scope: !7)
!41 = !DILocation(line: 44, column: 22, scope: !12, inlinedAt: !42)
!42 = !DILocation(line: 89, column: 33, scope: !7)
!43 = !DILocation(line: 44, column: 28, scope: !12, inlinedAt: !42)
!44 = !DILocation(line: 98, column: 33, scope: !7)
!45 = !DILocation(line: 89, column: 22, scope: !7)
!46 = !DILocation(line: 92, column: 51, scope: !7)
!47 = !DILocation(line: 92, column: 20, scope: !7)
!48 = !DILocation(line: 93, column: 20, scope: !7)
!49 = !DILocation(line: 97, column: 18, scope: !7)
!50 = !DILocation(line: 98, column: 18, scope: !7)
!51 = !DILocation(line: 92, column: 55, scope: !7)
!52 = !DILocation(line: 95, column: 33, scope: !7)
!53 = !DILocation(line: 109, column: 33, scope: !7)
!54 = !DILocation(line: 109, column: 21, scope: !7)
!55 = !DILocation(line: 109, column: 52, scope: !7)
!56 = !DILocation(line: 110, column: 33, scope: !7)
!57 = !DILocation(line: 110, column: 58, scope: !7)
!58 = !DILocation(line: 110, column: 39, scope: !7)
!59 = !DILocation(line: 111, column: 21, scope: !7)
!60 = !DILocation(line: 111, column: 4, scope: !7)
