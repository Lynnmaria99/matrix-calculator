# Release Notes

## 1.3.0 — 2026-11-16

### Desktop learning experience

- Added page-level guidance for matrix shapes, operation requirements, and built-in examples.
- Improved common input and calculus-domain error messages, including rejection of NaN and infinity matrix entries.
- Clarified floating-point display precision, rank thresholds, quadratic-form classification tolerance, and complex eigenvalue notation.
- Added condition-number and least-squares residual context to linear-system results.
- Included numerical notes and operation explanations when copying or exporting complete results as TXT.

### Project cleanup

- Retired the Streamlit application, its launch entry point, dependency, and dedicated tests.
- Kept the shared computation core, desktop app, command-line interface, and Notebook examples.
- Rebuilt the standalone Windows x64 package.

### 简体中文

- 桌面页面新增输入形状、运算要求和内置示例引导。
- 改进常见输入错误与微积分定义域提示；矩阵输入拒绝 NaN 和无穷值。
- 明确浮点显示精度、秩阈值、二次型分类容差及复数特征值表示。
- 线性方程组结果补充条件数与最小二乘残差说明。
- 复制结果和 TXT 导出包含数值说明及运算解释。
- 移除 Streamlit 应用、启动入口、依赖及专属测试，保留共享计算核心、桌面端、命令行和 Notebook 示例。
- 重新构建 Windows x64 独立版。
