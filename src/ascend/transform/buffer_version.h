/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tl/ascend/transform/buffer_version.h
 * \brief Annotation contract for buffer version metadata.
 */

#pragma once

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include "ascend/op/utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

/*! \brief Frontend overrides before AutoSchedule; selected versions afterward.
 */
constexpr const char *kBufferVersionsMap = "tl.buffer_versions_map";

/*! \brief Manual-version annotation and physical-version staging key. */
constexpr const char *kManualMultiBuffer = "tl.manual_multi_buffer";

/*! \brief Loop-local list of storage Vars eligible for automatic versioning. */
constexpr const char *kMultiBufferEligible = "multi_buffer_eligible";

/*! \brief Storage data Var -> positive compile-time version count. */
using BufferVersionMap = ffi::Map<tirx::Var, int>;

// Selected physical L0B reuse pairs. Layouts and logical buffers remain
// distinct, but the selected pair advances one storage-ring protocol.
constexpr const char *kReuseRingBindings = "tl.reuse_ring_bindings";
using ReuseRingBindings = ffi::Map<tirx::Var, tirx::Var>;

// Logical reads/writes remain separate. Ownership, physical versions and
// address reuse use the same allocation group, represented by the data Var.
class L0StorageGroups {
public:
  explicit L0StorageGroups(L0SFBindings bindings = {},
                           ReuseRingBindings reuse_bindings = {})
      : sf_bindings_(std::move(bindings)),
        reuse_bindings_(std::move(reuse_bindings)), bindings_(sf_bindings_) {
    for (const auto &[member, data] : reuse_bindings_)
      bindings_.Set(member, data);
    for (const auto &[member, data] : bindings_) {
      auto members = members_.Get(data).value_or(ffi::Array<tirx::Var>{data});
      members.push_back(member);
      members_.Set(data, members);
    }
  }
  tirx::Var Representative(const tirx::Var &storage) const {
    return bindings_.Get(storage).value_or(storage);
  }
  ffi::Array<tirx::Var> Members(const tirx::Var &storage) const {
    tirx::Var data = Representative(storage);
    return members_.Get(data).value_or(ffi::Array<tirx::Var>{data});
  }
  const L0SFBindings &Bindings() const { return sf_bindings_; }
  bool SharesReuseRing(const tirx::Var &lhs, const tirx::Var &rhs) const {
    auto lhs_data = reuse_bindings_.Get(lhs);
    auto rhs_data = reuse_bindings_.Get(rhs);
    return (lhs_data.has_value() && lhs_data.value().same_as(rhs)) ||
           (rhs_data.has_value() && rhs_data.value().same_as(lhs));
  }

private:
  L0SFBindings sf_bindings_;
  ReuseRingBindings reuse_bindings_;
  L0SFBindings bindings_;
  ffi::Map<tirx::Var, ffi::Array<tirx::Var>> members_;
};

inline L0StorageGroups CollectL0StorageGroups(const tirx::SBlock &root) {
  ReuseRingBindings reuse;
  if (auto value = root->annotations.Get(kReuseRingBindings))
    reuse = value.value().cast<ReuseRingBindings>();
  return L0StorageGroups(CollectL0SFBindings(root), std::move(reuse));
}

// User overrides and prepared metadata must describe the entire group.
// Contradictory member requests are errors, rather than last-writer wins.
template <typename T>
ffi::Map<tirx::Var, T> ExpandL0StorageGroupValues(ffi::Map<tirx::Var, T> values,
                                                  const L0StorageGroups &groups,
                                                  const char *description) {
  auto original = values;
  for (const auto &[storage, value] : original) {
    for (const tirx::Var &member : groups.Members(storage)) {
      if (auto existing = values.Get(member)) {
        ICHECK(existing.value() == value)
            << "Conflicting " << description << " for bound L0 data/SF group "
            << groups.Representative(storage)->name_hint;
      }
      values.Set(member, value);
    }
  }
  return values;
}

} // namespace tl
} // namespace tvm
