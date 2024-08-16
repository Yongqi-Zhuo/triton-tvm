#ifndef TVM_TYPES
#define TVM_TYPES

include "mlir/IR/AttrTypeBase.td"

include "triton-tvm/Dialect/TVM/IR/Dialect.td"


class TVM_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TVM_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif
