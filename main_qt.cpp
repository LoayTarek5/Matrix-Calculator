#include "mainwindow.h"

#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QRadioButton>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QMessageBox>
#include <QSpinBox>
#include <QStyleFactory>
#include <QTextEdit>
#include <QInputDialog>
#include <QComboBox>
#include <QStyledItemDelegate>
#include <QRegExp>
#include <QGraphicsDropShadowEffect>
#include<Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <bits/stdc++.h>
using namespace std;
using namespace Eigen;

// Matrix Operations (from the original code)
class MatrixOperations {
public:
    // Function to create matrix (from original code)
    static vector<vector<double>> createMatrix(int rows, int cols, const vector<double>& values) {
        vector<vector<double>> matrix;
        for (int i = 0; i < rows; i++) {
            vector<double> row;
            for (int j = 0; j < cols; j++) {
                row.push_back(values[i * cols + j]);
            }
            matrix.push_back(row);
        }
        return matrix;
    }

    // Transpose matrix (from original code)
    static vector<vector<double>> matrixTranspose(const vector<vector<double>>& mat) {
        vector<vector<double>> cont;
        for (int i = 0; i < mat[0].size(); i++) {
            vector<double> vec;
            for (int j = 0; j < mat.size(); j++) {
                vec.push_back(mat[j][i]);
            }
            cont.push_back(vec);
        }
        return cont;
    }

    // Multiply matrices (from original code)
    static vector<vector<double>> multiplyMatrix(
        const vector<vector<double>>& m1,
        const vector<vector<double>>& m2) {
        vector<vector<double>> cont;
        if (m1[0].size() != m2.size()) {
            return cont;
        }
        auto m2_transposed = matrixTranspose(m2);
        for (int i = 0; i < m1.size(); i++) {
            vector<double> vec;
            for (int j = 0; j < m2[0].size(); j++) {
                double sum = 0;
                for (int u = 0; u < m1[0].size(); u++) {
                    sum += m1[i][u] * m2_transposed[j][u];
                }
                vec.push_back(sum);
            }
            cont.push_back(vec);
        }
        return cont;
    }

    // Add matrices (from original code)
    static vector<vector<double>> addMatrix(
        const vector<vector<double>>& m1,
        const vector<vector<double>>& m2) {
        if (m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
            return {};
        }
        vector<vector<double>> result = m1;
        for (int i = 0; i < m1.size(); i++) {
            for (int j = 0; j < m1[0].size(); j++) {
                result[i][j] += m2[i][j];
            }
        }
        return result;
    }

    // Subtract matrices (from original code)
    static vector<vector<double>> subtractMatrix(
        const vector<vector<double>>& m1,
        const vector<vector<double>>& m2) {
        if (m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
            return {};
        }
        vector<vector<double>> result = m1;
        for (int i = 0; i < m1.size(); i++) {
            for (int j = 0; j < m1[0].size(); j++) {
                result[i][j] -= m2[i][j];
            }
        }
        return result;
    }

    // Determinant calculation (from original code)
    static double matrixDeterminant(vector<vector<double>> matrix) {
        int n = matrix.size();
        double det = 1.0;
        int swapCount = 0;

        for (int i = 0; i < n; ++i) {
            int pivotRow = i;
            for (int j = i + 1; j < n; ++j) {
                if (fabs(matrix[j][i]) > fabs(matrix[pivotRow][i])) {
                    pivotRow = j;
                }
            }

            if (i != pivotRow) {
                swap(matrix[i], matrix[pivotRow]);
                swapCount++;
            }

            if (fabs(matrix[i][i]) < 1e-9) {
                return 0.0;
            }

            for (int j = i + 1; j < n; ++j) {
                double factor = matrix[j][i] / matrix[i][i];
                for (int k = i; k < n; ++k) {
                    matrix[j][k] -= factor * matrix[i][k];
                }
            }

            det *= matrix[i][i];
        }

        if (swapCount % 2 != 0) {
            det = -det;
        }

        return det;
    }

    // Solve system of equations (from original code)
    static vector<double> solveSystem(
        const vector<vector<double>>& LHS,
        const vector<double>& RHS) {
        double det = matrixDeterminant(LHS);
        if (det == 0) {
            return {};
        }
        int matsize = LHS.size();
        vector<double> ans;
        auto LHS_transposed = matrixTranspose(LHS);
        for (int i = 0; i < matsize; i++) {
            vector<vector<double>> newvec = LHS_transposed;
            newvec[i] = RHS;
            newvec = matrixTranspose(newvec);
            ans.push_back(matrixDeterminant(newvec) / det);
        }
        return ans;
    }

};

// Additional Matrix Operations
class AdvancedMatrixOperations {
public:
    // LU Decomposition using Doolittle method
    static pair<vector<vector<double>>, vector<vector<double>>> doolittleLUDecomposition(
        const vector<vector<double>>& matrix) {
        int n = matrix.size();
        vector<vector<double>> L(n, vector<double>(n, 0));
        vector<vector<double>> U(n, vector<double>(n, 0));

        for (int i = 0; i < n; i++) {
            // Upper triangular matrix
            for (int k = i; k < n; k++) {
                double sum = 0;
                for (int j = 0; j < i; j++) {
                    sum += L[i][j] * U[j][k];
                }
                U[i][k] = matrix[i][k] - sum;
            }

            // Lower triangular matrix
            L[i][i] = 1;
            for (int k = i + 1; k < n; k++) {
                double sum = 0;
                for (int j = 0; j < i; j++) {
                    sum += L[k][j] * U[j][i];
                }
                L[k][i] = (matrix[k][i] - sum) / U[i][i];
            }
        }

        return {L, U};
    }

    static pair<vector<vector<double>>, vector<vector<double>>> crout_decomposition(const vector<vector<double>> &matrix) {
        int n = matrix.size();
        vector<vector<double>> L(n, vector<double>(n, 0));
        vector<vector<double>> U(n, vector<double>(n, 0));

        for (int i = 0; i < n; i++) {
            U[i][i] = 1;
        }

        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
                double sum = 0;
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * U[k][j];
                }
                L[i][j] = matrix[i][j] - sum;
            }

            for (int i = j + 1; i < n; i++) {
                double sum = 0;
                for (int k = 0; k < j; k++) {
                    sum += L[j][k] * U[k][i];
                }
                U[j][i] = (matrix[j][i] - sum) / L[j][j];
            }
        }

        return {L, U};
    }

    // Matrix Inverse using Gaussian Elimination
    static vector<vector<double>> matrixInverse(const vector<vector<double>>& matrix) {
        int n = matrix.size();

        // Check if matrix is square
        if (n == 0 || n != matrix[0].size()) {
            throw runtime_error("Matrix must be square for inverse calculation.");
        }

        // Compute determinant first
        double det = MatrixOperations::matrixDeterminant(matrix);
        if (abs(det) < 1e-10) {
            throw runtime_error("Matrix is singular, inverse does not exist.");
        }

        // Create an augmented matrix [A|I]
        vector<vector<double>> augmented(n, vector<double>(2*n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
                augmented[i][j+n] = (i == j) ? 1.0 : 0.0;
            }
        }

        // Gaussian elimination
        for (int i = 0; i < n; i++) {
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (abs(augmented[k][i]) > abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }

            swap(augmented[i], augmented[maxRow]);

            double pivot = augmented[i][i];
            for (int j = i; j < 2*n; j++) {
                augmented[i][j] /= pivot;
            }

            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = i; j < 2*n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }

        // Extract inverse
        vector<vector<double>> inverse(n, vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][j+n];
            }
        }

        return inverse;
    }

    // Power Method for dominant eigenvalue
    static pair<double, vector<double>> powerMethod(
        const vector<vector<double>>& matrix,
        int maxIterations = 100,
        double tolerance = 1e-10) {
        int n = matrix.size();

        // Check square matrix
        if (n == 0 || n != matrix[0].size()) {
            throw runtime_error("Matrix must be square for eigenvalue calculation.");
        }

        vector<double> v(n, 1.0);
        double eigenvalue = 0;

        for (int iter = 0; iter < maxIterations; iter++) {
            // Multiply matrix by vector
            vector<double> y(n, 0);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    y[i] += matrix[i][j] * v[j];
                }
            }

            // Normalize vector
            double norm = 0;
            for (double val : y) {
                norm += val * val;
            }
            norm = sqrt(norm);

            for (double &val : y) {
                val /= norm;
            }

            // Compute Rayleigh quotient
            double newEigenvalue = 0;
            for (int i = 0; i < n; i++) {
                double dot = 0;
                for (int j = 0; j < n; j++) {
                    dot += matrix[i][j] * y[j];
                }
                newEigenvalue += dot * y[i];
            }

            // Check convergence
            if (abs(newEigenvalue - eigenvalue) < tolerance) {
                return {newEigenvalue, y};
            }

            eigenvalue = newEigenvalue;
            v = y;
        }

        return {eigenvalue, v};
    }

    static double calculateTrace(const vector<vector<double>>& matrix) {
        if (matrix.empty() || matrix.size() != matrix[0].size()) {
            throw runtime_error("Invalid matrix for trace calculation");
        }

        Eigen::MatrixXd eigenMatrix = convertToEigenMatrix(matrix);
        return eigenMatrix.trace();
    }

    static vector<vector<double>> calculateMatrixPower(const vector<vector<double>>& matrix, int power) {
        if (power < 0) {
            throw runtime_error("Negative matrix power is not supported");
        }

        Eigen::MatrixXd eigenMatrix = convertToEigenMatrix(matrix);

        // Special cases
        if (power == 0) {
            // Return identity matrix of the same size
            Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(eigenMatrix.rows(), eigenMatrix.cols());
            return convertToStdVector(identityMatrix);
        }

        if (power == 1) {
            return convertToStdVector(eigenMatrix);
        }

        // For power > 1, use matrix multiplication
        Eigen::MatrixXd resultMatrix = eigenMatrix;
        for (int i = 1; i < power; ++i) {
            resultMatrix *= eigenMatrix;
        }

        return convertToStdVector(resultMatrix);
    }

    static vector<vector<double>> scalarMultiplyMatrix(const vector<vector<double>>& matrix, double scalar) {
        Eigen::MatrixXd eigenMatrix = convertToEigenMatrix(matrix);
        Eigen::MatrixXd resultMatrix = eigenMatrix * scalar;
        return convertToStdVector(resultMatrix);
    }

    static int calculateMatrixRank(const vector<vector<double>>& matrix) {
        Eigen::MatrixXd eigenMatrix = convertToEigenMatrix(matrix);
        Eigen::FullPivLU<Eigen::MatrixXd> fullPivLU(eigenMatrix);
        return fullPivLU.rank();
    }

    static pair<vector<vector<double>>, vector<vector<double>>> choleskyDecomposition(const vector<vector<double>>& matrix) {
        // Helper function to check if a matrix is square
        auto isSquare = [](const vector<vector<double>>& mat) {
            size_t rows = mat.size();
            for (const auto& row : mat) {
                if (row.size() != rows) {
                    return false;
                }
            }
            return true;
        };

        // Helper function to check if a matrix is symmetric
        auto isSymmetric = [](const vector<vector<double>>& mat) {
            size_t n = mat.size();
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    if (mat[i][j] != mat[j][i]) {
                        return false;
                    }
                }
            }
            return true;
        };

        // Check if the matrix is square
        if (!isSquare(matrix)) {
            throw runtime_error("Matrix is not square.");
        }

        // Check if the matrix is symmetric
        if (!isSymmetric(matrix)) {
            throw runtime_error("Matrix is not symmetric.");
        }

        // Convert to Eigen Matrix
        Eigen::MatrixXd eigenMatrix = convertToEigenMatrix(matrix);

        // Perform Cholesky decomposition using Eigen
        Eigen::LLT<Eigen::MatrixXd> llt(eigenMatrix);

        // Check if the matrix is positive definite
        if (llt.info() != Eigen::Success) {
            throw runtime_error("Matrix is not positive definite.");
        }

        // Extract L and U matrices
        Eigen::MatrixXd L = llt.matrixL();
        Eigen::MatrixXd U = llt.matrixU();

        // Convert Eigen matrices back to std::vector and return
        return {convertToStdVector(L), convertToStdVector(U)};
    }

    static double calculateNorm(const vector<vector<double>>& matrix, int normType = 0) {
        Eigen::MatrixXd eigenMatrix = convertToEigenMatrix(matrix);

        switch(normType) {
        case 0: // Frobenius Norm (default)
            return eigenMatrix.norm();
        case 1: // 1-Norm (max column sum)
            return eigenMatrix.cwiseAbs().colwise().sum().maxCoeff();
        case 2: // Infinity Norm (max row sum)
            return eigenMatrix.cwiseAbs().rowwise().sum().maxCoeff();
        default:
            throw runtime_error("Invalid norm type");
        }
    }

    // Helper methods for conversion between vector and Eigen::Matrix
    static Eigen::MatrixXd convertToEigenMatrix(const vector<vector<double>>& matrix) {
        if (matrix.empty()) {
            throw runtime_error("Empty matrix");
        }

        Eigen::MatrixXd eigenMatrix(matrix.size(), matrix[0].size());
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                eigenMatrix(i, j) = matrix[i][j];
            }
        }
        return eigenMatrix;
    }

    static vector<vector<double>> convertToStdVector(const Eigen::MatrixXd& matrix) {
        vector<vector<double>> stdMatrix(matrix.rows(), vector<double>(matrix.cols()));
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                stdMatrix[i][j] = matrix(i, j);
            }
        }
        return stdMatrix;
    }

    static vector<vector<double>> performMatrixArithmeticOperation(
        const vector<vector<double>>& matrix1,
        const vector<vector<double>>& matrix2,
        const string& operation) {
        // Validate matrix dimensions
        if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size()) {
            throw runtime_error("Matrices must have the same dimensions");
        }

        Eigen::MatrixXd eigenMatrix1 = convertToEigenMatrix(matrix1);
        Eigen::MatrixXd eigenMatrix2 = convertToEigenMatrix(matrix2);

        Eigen::MatrixXd resultMatrix;

        // Parse the operation
        QRegExp operationPattern(R"((\-?\d+\.?\d*)\s*\*?\s*([AB])\s*([\+\-])\s*(\-?\d+\.?\d*)\s*\*?\s*([AB]))");

        if (operationPattern.exactMatch(QString::fromStdString(operation))) {
            double coeff1 = operationPattern.cap(1).toDouble();
            char matrix1Var = operationPattern.cap(2).toStdString()[0];
            char operatorSymbol = operationPattern.cap(3).toStdString()[0];
            double coeff2 = operationPattern.cap(4).toDouble();
            char matrix2Var = operationPattern.cap(5).toStdString()[0];

            Eigen::MatrixXd selectedMatrix1 = (matrix1Var == 'A') ? eigenMatrix1 : eigenMatrix2;
            Eigen::MatrixXd selectedMatrix2 = (matrix2Var == 'B') ? eigenMatrix2 : eigenMatrix1;

            // Perform the operation based on operator
            switch (operatorSymbol) {
            case '+':
                resultMatrix = coeff1 * selectedMatrix1 + coeff2 * selectedMatrix2;
                break;
            case '-':
                resultMatrix = coeff1 * selectedMatrix1 - coeff2 * selectedMatrix2;
                break;
            default:
                throw runtime_error("Unsupported operation");
            }
        } else {
            throw runtime_error("Invalid operation format");
        }

        return convertToStdVector(resultMatrix);
    }

    static Eigen::MatrixXd gramSchmidt(const Eigen::MatrixXd& input) {
        int rows = input.rows();
        int cols = input.cols();
        Eigen::MatrixXd orthonormalBasis(rows, cols);
        orthonormalBasis.setZero(); // Initialize the matrix with zeros

        for (int i = 0; i < cols; ++i) {
            Eigen::VectorXd v = input.col(i); // Start with the i-th vector

            // Subtract the projections onto the previously computed basis vectors
            for (int j = 0; j < i; ++j) {
                Eigen::VectorXd u = orthonormalBasis.col(j);
                v -= (u.dot(v) / u.dot(u)) * u;
            }

            // Normalize the vector
            double norm = v.norm();
            if (norm > 1e-10) { // Avoid division by zero for degenerate cases
                orthonormalBasis.col(i) = v / norm;
            } else {
                cerr << "Linearly dependent or near-zero vector detected at column " << i << endl;
            }
        }

        return orthonormalBasis;
    }

    static Eigen::MatrixXd computeAdjoint(const Eigen::MatrixXd& matrix) {
        int n = matrix.rows();
        if (n != matrix.cols()) {
            throw invalid_argument("Matrix must be square to compute adjoint.");
        }

        Eigen::MatrixXd adjoint(n, n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // Compute the minor matrix by removing row i and column j
                Eigen::MatrixXd minor(n - 1, n - 1);

                // Copy rows and columns except the i-th row and j-th column
                int rowOffset = 0;
                for (int r = 0; r < n - 1; ++r) {
                    if (r == i) rowOffset = 1;
                    int colOffset = 0;
                    for (int c = 0; c < n - 1; ++c) {
                        if (c == j) colOffset = 1;
                        minor(r, c) = matrix(r + rowOffset, c + colOffset);
                    }
                }

                // Compute the determinant of the minor matrix
                double determinant = minor.determinant();

                // Assign the cofactor (with the alternating sign)
                // Note: we transpose here to get the adjoint
                adjoint(j, i) = (i + j) % 2 == 0 ? determinant : -determinant;
            }
        }

        return adjoint;
    }

};

class ComplexDelegate : public QStyledItemDelegate {
public:
    QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option,
                          const QModelIndex& index) const override {
        QLineEdit* editor = new QLineEdit(parent);
        QRegularExpression complexRegex(
            R"(^([+-]?\d*\.?\d*)\s*([+-])?\s*([+-]?\d*\.?\d*)?i?$)"
            );
        QRegularExpressionValidator* validator =
            new QRegularExpressionValidator(complexRegex, editor);
        editor->setValidator(validator);
        return editor;
    }

    void setEditorData(QWidget* editor, const QModelIndex& index) const override {
        QString value = index.model()->data(index, Qt::EditRole).toString();
        QLineEdit* lineEdit = qobject_cast<QLineEdit*>(editor);
        if (lineEdit) {
            lineEdit->setText(value);
        }
    }

    void setModelData(QWidget* editor, QAbstractItemModel* model,
                      const QModelIndex& index) const override {
        QLineEdit* lineEdit = qobject_cast<QLineEdit*>(editor);
        if (lineEdit) {
            QString text = lineEdit->text().trimmed();
            model->setData(index, text, Qt::EditRole);
        }
    }
};

class MatrixCalculatorGUI : public QMainWindow {
    Q_OBJECT

public:
    MatrixCalculatorGUI(QWidget *parent = nullptr) : QMainWindow(parent) {

        setWindowTitle("FCAI Students under supervisor Dr. Eng. Moustafa Reda A. Eltantawi MA214-Math-3");
        setMinimumSize(1000, 700);

        // Central widget and main layout
        QWidget *centralWidget = new QWidget(this);
        QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

        // Matrix Resize Section
        QHBoxLayout *resizeLayout = new QHBoxLayout();

        // Matrix 1 Resize Controls
        QVBoxLayout *matrix1ResizeLayout = new QVBoxLayout();
        QLabel *matrix1ResizeLabel = new QLabel("Matrix 1 Dimensions:");
        QHBoxLayout *matrix1DimLayout = new QHBoxLayout();
        QLabel *matrix1RowsLabel = new QLabel("Rows:");
        matrix1RowsSpin = new QSpinBox();
        matrix1RowsSpin->setRange(1, 100);
        matrix1RowsSpin->setValue(3);
        QLabel *matrix1ColsLabel = new QLabel("Columns:");
        matrix1ColsSpin = new QSpinBox();
        matrix1ColsSpin->setRange(1, 100);
        matrix1ColsSpin->setValue(3);
        matrix1DimLayout->addWidget(matrix1RowsLabel);
        matrix1DimLayout->addWidget(matrix1RowsSpin);
        matrix1DimLayout->addWidget(matrix1ColsLabel);
        matrix1DimLayout->addWidget(matrix1ColsSpin);
        matrix1ResizeLayout->addWidget(matrix1ResizeLabel);
        matrix1ResizeLayout->addLayout(matrix1DimLayout);

        // Matrix 2 Resize Controls
        QVBoxLayout *matrix2ResizeLayout = new QVBoxLayout();
        QLabel *matrix2ResizeLabel = new QLabel("Matrix 2 Dimensions:");
        QHBoxLayout *matrix2DimLayout = new QHBoxLayout();
        QLabel *matrix2RowsLabel = new QLabel("Rows:");
        matrix2RowsSpin = new QSpinBox();
        matrix2RowsSpin->setRange(1, 100);
        matrix2RowsSpin->setValue(3);
        QLabel *matrix2ColsLabel = new QLabel("Columns:");
        matrix2ColsSpin = new QSpinBox();
        matrix2ColsSpin->setRange(1, 100);
        matrix2ColsSpin->setValue(3);
        matrix2DimLayout->addWidget(matrix2RowsLabel);
        matrix2DimLayout->addWidget(matrix2RowsSpin);
        matrix2DimLayout->addWidget(matrix2ColsLabel);
        matrix2DimLayout->addWidget(matrix2ColsSpin);
        matrix2ResizeLayout->addWidget(matrix2ResizeLabel);
        matrix2ResizeLayout->addLayout(matrix2DimLayout);

        // Resize Buttons
        QPushButton *resize1Button = new QPushButton("Resize Matrix 1");
        QPushButton *resize2Button = new QPushButton("Resize Matrix 2");
        matrix1ResizeLayout->addWidget(resize1Button);
        matrix2ResizeLayout->addWidget(resize2Button);

        resizeLayout->addLayout(matrix1ResizeLayout);
        resizeLayout->addLayout(matrix2ResizeLayout);
        mainLayout->addLayout(resizeLayout);

        // Matrix input sections
        QHBoxLayout *matricesLayout = new QHBoxLayout();

        // First Matrix
        QVBoxLayout *matrix1Layout = new QVBoxLayout();
        QLabel *matrix1Label = new QLabel("Matrix 1:");
        matrix1Table = new QTableWidget(3, 3);
        matrix1Table->setMinimumSize(250, 200);
        matrix1Layout->addWidget(matrix1Label);
        matrix1Layout->addWidget(matrix1Table);

        // Second Matrix
        QVBoxLayout *matrix2Layout = new QVBoxLayout();
        QLabel *matrix2Label = new QLabel("Matrix 2:");
        matrix2Table = new QTableWidget(3, 3);
        matrix2Table->setMinimumSize(250, 200);
        matrix2Layout->addWidget(matrix2Label);
        matrix2Layout->addWidget(matrix2Table);

        // Result Matrix
        QVBoxLayout *resultLayout = new QVBoxLayout();
        QLabel *resultLabel = new QLabel("Result:");
        resultTable = new QTableWidget(3, 3);
        resultTable->setMinimumSize(250, 200);
        resultTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        resultLayout->addWidget(resultLabel);
        resultLayout->addWidget(resultTable);

        // Add matrix inputs to main layout
        matricesLayout->addLayout(matrix1Layout);
        matricesLayout->addLayout(matrix2Layout);
        matricesLayout->addLayout(resultLayout);
        mainLayout->addLayout(matricesLayout);

        // Operation Buttons
        QHBoxLayout *operationLayout = new QHBoxLayout();
        QPushButton *addButton = new QPushButton("Add");
        QPushButton *subtractButton = new QPushButton("Subtract");
        QPushButton *multiplyButton = new QPushButton("Multiply");
        QPushButton *transposeButton = new QPushButton("Transpose");
        QPushButton *determinantButton = new QPushButton("Determinant");
        QPushButton *insertToMatrix1Button = new QPushButton("Insert to Matrix 1");
        QPushButton *insertToMatrix2Button = new QPushButton("Insert to Matrix 2");

        // addButton->minimumHeight();

        // Add new tab widget for advanced operations
        QTabWidget *operationTabs = new QTabWidget();
        operationTabs->setStyleSheet(R"(
        QTabWidget::pane {
            border: 1px solid #d3d3d3;
            background: white;
        }
        QTabBar::tab {
            background: #f0f0f0;
            color: #333;
            padding: 8px 16px;
            margin-right: 4px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #e0e0e0;
            border-bottom: 2px solid #007bff;
            color: #007bff;
        }
        QTabBar::tab:hover {
            background: #e6e6e6;
        }
    )");



        // Dark theme styling
        setStyleSheet(
            "QTableWidget { "
            "   background-color: #1e1e1e; "
            "   gridline-color: #404040; "
            "   color: #d4d4d4; "
            "   border: 1px solid #404040; "
            "} "
            );

        // Set background color for the entire panel
        QPalette pal = palette();
        pal.setColor(QPalette::Window, QColor("#252526"));
        setAutoFillBackground(true);
        setPalette(pal);
        QString globalStyleSheet = R"(
    QMainWindow { background-color: #1e1e1e; }
    QWidget { color: #d4d4d4; background-color: #252526; }
    QStatusBar { background-color: #007acc; color: white; }
    QPushButton {
        background-color: #0e639c;
        color: #ffffff;
        border: none;
        min-height: 25px;
        border-radius: 3px;
    }
    QPushButton:hover { background-color: #1177bb; }
    QPushButton:pressed { background-color: #094771; }
    QPushButton:disabled {
        background-color: #2d2d2d;
        color: #808080;
    }
    QLineEdit {
        background-color: #1e1e1e;
        color: #d4d4d4;
        border: 1px solid #404040;
        padding: 2px;
    }
    QTabWidget::pane {
        border: 1px solid #d3d3d3;
        background: white;
    }
    QTabBar::tab {
        background: #f0f0f0;
        color: #333;
        padding: 8px 16px;
        margin-right: 4px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background: #e0e0e0;
        border-bottom: 2px solid #007bff;
        color: #007bff;
    }
    QTabBar::tab:hover { background: #e6e6e6; }
)";
        qApp->setStyle(QStyleFactory::create("Fusion"));
        qApp->setStyleSheet(globalStyleSheet);

        // Dark theme styling for buttons
        QString buttonStyle =
            "QPushButton { "
            "   background-color: #0e639c; "
            "   color: #ffffff; "
            "   border: none; "
            " min-height: 25px; "
            " min-width: fit-content; "
            "   border-radius: 3px; "
            "} "
            "QPushButton:hover { "
            "   background-color: #1177bb; "
            "} "
            "QPushButton:pressed { "
            "   background-color: #094771; "
            "} "
            "QPushButton:disabled { "
            "   background-color: #2d2d2d; "
            "   color: #808080; "
            "}";

        // Dark theme styling for labels
        QString labelStyle =
            "QLabel { "
            "   color: #d4d4d4; "
            "}";

        // Dark theme styling for input fields
        QString inputStyle =
            "QLineEdit { "
            "   background-color: #1e1e1e; "
            "   color: #d4d4d4; "
            "   border: 1px solid #404040; "
            "   padding: 2px; "
            "} "
            "QLineEdit:disabled { "
            "   background-color: #2d2d2d; "
            "   color: #808080; "
            "}";

        // Apply styles to components
        setStyleSheet(inputStyle + buttonStyle + labelStyle);

        // Create icons for tabs
        QIcon basicIcon = style()->standardIcon(QStyle::SP_ComputerIcon);
        QIcon advancedIcon = style()->standardIcon(QStyle::SP_FileDialogDetailedView);
        QIcon singleMatrixIcon = style()->standardIcon(QStyle::SP_FileDialogContentsView);

        // Basic Single Matrix Operations Tab
        QWidget *singleMatrixOpsWidget = new QWidget();
        QVBoxLayout *singleMatrixOpsLayout = new QVBoxLayout(singleMatrixOpsWidget);
        singleMatrixOpsLayout->setSpacing(10);
        singleMatrixOpsLayout->setContentsMargins(10, 10, 10, 10);


        // Matrix 1 Operations
        QLabel *matrix1OpsLabel = new QLabel("Matrix 1 Operations:");
        matrix1OpsLabel->setStyleSheet("font-weight: bold; color: #FFF;");
        QHBoxLayout *matrix1OpsLayout = new QHBoxLayout();

        QString paddStlye = "QPushButton {"
                            // "margin-left: 5px;"
                            // "margin-right: 5px;"
                            "padding: 5px;"
                            // " min-width: 90px; "
                            "}";

        QPushButton *matrix1TransposeButton = new QPushButton("Transpose");
        matrix1TransposeButton->setStyleSheet(paddStlye);
        QPushButton *matrix1DeterminantButton = new QPushButton("Determinant");
        matrix1DeterminantButton->setStyleSheet(paddStlye);
        QPushButton *matrix1InverseButton = new QPushButton("Inverse");
        matrix1InverseButton->setStyleSheet(paddStlye);
        QPushButton *matrix1LUDecompButton = new QPushButton("LU Decomposition");
        matrix1LUDecompButton->setStyleSheet(paddStlye);
        QPushButton *matrix1EigenvalueButton = new QPushButton("Eigen values And Vectors");
        matrix1EigenvalueButton->setStyleSheet(paddStlye);
        QPushButton *matrix1TraceButton = new QPushButton("Trace");
        matrix1TraceButton->setStyleSheet(paddStlye);
        QPushButton *matrix1PowerButton = new QPushButton("Matrix Power");
        matrix1PowerButton->setStyleSheet(paddStlye);
        QPushButton *matrix1ScalarMultiplyButton = new QPushButton("Scalar Multiply");
        matrix1ScalarMultiplyButton->setStyleSheet(paddStlye);
        QPushButton *matrix1RankButton = new QPushButton("Rank");
        matrix1RankButton->setStyleSheet(paddStlye);
        QPushButton *matrix1CholeskyButton = new QPushButton("Cholesky Decomposition");
        matrix1CholeskyButton->setStyleSheet(paddStlye);
        QPushButton *matrix1NormButton = new QPushButton("Matrix Norm");
        matrix1NormButton->setStyleSheet(paddStlye);
        QPushButton *matrix1GramSchmidtButton = new QPushButton("Gram-Schmidt");
        matrix1GramSchmidtButton->setStyleSheet(paddStlye);
        QPushButton *matrix1ConjugateButton = new QPushButton("Conjugate");
        matrix1ConjugateButton->setStyleSheet(paddStlye);
        QPushButton *matrix1AdjointButton = new QPushButton("Adjoint");
        matrix1AdjointButton->setStyleSheet(paddStlye);

        matrix1OpsLayout->addWidget(matrix1ConjugateButton);
        matrix1OpsLayout->addWidget(matrix1AdjointButton);
        matrix1OpsLayout->addWidget(matrix1GramSchmidtButton);
        matrix1OpsLayout->addWidget(matrix1TransposeButton);
        matrix1OpsLayout->addWidget(matrix1DeterminantButton);
        matrix1OpsLayout->addWidget(matrix1InverseButton);
        matrix1OpsLayout->addWidget(matrix1LUDecompButton);
        matrix1OpsLayout->addWidget(matrix1EigenvalueButton);
        matrix1OpsLayout->addWidget(matrix1TraceButton);
        matrix1OpsLayout->addWidget(matrix1PowerButton);
        matrix1OpsLayout->addWidget(matrix1ScalarMultiplyButton);
        matrix1OpsLayout->addWidget(matrix1RankButton);
        matrix1OpsLayout->addWidget(matrix1CholeskyButton);
        matrix1OpsLayout->addWidget(matrix1NormButton);

        // Matrix 2 Operations
        QLabel *matrix2OpsLabel = new QLabel("Matrix 2 Operations:");
        matrix2OpsLabel->setStyleSheet("font-weight: bold; color: #FFF;");

        QHBoxLayout *matrix2OpsLayout = new QHBoxLayout();
        QPushButton *matrix2TransposeButton = new QPushButton("Transpose");
        matrix2TransposeButton->setStyleSheet(paddStlye);
        QPushButton *matrix2DeterminantButton = new QPushButton("Determinant");
        matrix2DeterminantButton->setStyleSheet(paddStlye);
        QPushButton *matrix2InverseButton = new QPushButton("Inverse");
        matrix2InverseButton->setStyleSheet(paddStlye);
        QPushButton *matrix2LUDecompButton = new QPushButton("LU Decomposition");
        matrix2LUDecompButton->setStyleSheet(paddStlye);
        QPushButton *matrix2EigenvalueButton = new QPushButton("Eigen values And Vectors");
        matrix2EigenvalueButton->setStyleSheet(paddStlye);
        QPushButton *matrix2TraceButton = new QPushButton("Trace");
        matrix2TraceButton->setStyleSheet(paddStlye);
        QPushButton *matrix2PowerButton = new QPushButton("Matrix Power");
        matrix2PowerButton->setStyleSheet(paddStlye);
        QPushButton *matrix2ScalarMultiplyButton = new QPushButton("Scalar Multiply");
        matrix2ScalarMultiplyButton->setStyleSheet(paddStlye);
        QPushButton *matrix2RankButton = new QPushButton("Rank");
        matrix2RankButton->setStyleSheet(paddStlye);
        QPushButton *matrix2CholeskyButton = new QPushButton("Cholesky Decomposition");
        matrix2CholeskyButton->setStyleSheet(paddStlye);
        QPushButton *matrix2NormButton = new QPushButton("Matrix Norm");
        matrix2NormButton->setStyleSheet(paddStlye);
        QPushButton *matrix2GramSchmidtButton = new QPushButton("Gram-Schmidt");
        matrix2GramSchmidtButton->setStyleSheet(paddStlye);
        QPushButton *matrix2ConjugateButton = new QPushButton("Conjugate");
        matrix2ConjugateButton->setStyleSheet(paddStlye);
        QPushButton *matrix2AdjointButton = new QPushButton("Adjoint");
        matrix2AdjointButton->setStyleSheet(paddStlye);

        matrix2OpsLayout->addWidget(matrix2ConjugateButton);
        matrix2OpsLayout->addWidget(matrix2AdjointButton);
        matrix2OpsLayout->addWidget(matrix2GramSchmidtButton);
        matrix2OpsLayout->addWidget(matrix2TransposeButton);
        matrix2OpsLayout->addWidget(matrix2DeterminantButton);
        matrix2OpsLayout->addWidget(matrix2InverseButton);
        matrix2OpsLayout->addWidget(matrix2LUDecompButton);
        matrix2OpsLayout->addWidget(matrix2EigenvalueButton);
        matrix2OpsLayout->addWidget(matrix2TraceButton);
        matrix2OpsLayout->addWidget(matrix2PowerButton);
        matrix2OpsLayout->addWidget(matrix2ScalarMultiplyButton);
        matrix2OpsLayout->addWidget(matrix2RankButton);
        matrix2OpsLayout->addWidget(matrix2CholeskyButton);
        matrix2OpsLayout->addWidget(matrix2NormButton);

        matrix2OpsLayout->setSpacing(10);
        // Add to single matrix operations layout
        singleMatrixOpsLayout->addWidget(matrix1OpsLabel);
        singleMatrixOpsLayout->addLayout(matrix1OpsLayout);
        singleMatrixOpsLayout->addWidget(matrix2OpsLabel);
        singleMatrixOpsLayout->addLayout(matrix2OpsLayout);


        // Basic Operations Tab
        QWidget *basicOperationsWidget = new QWidget();
        QVBoxLayout *basicOperationLayout = new QVBoxLayout(basicOperationsWidget);
        basicOperationLayout->setSpacing(10);
        basicOperationLayout->setContentsMargins(10, 10, 10, 10);

        basicOperationLayout->addWidget(addButton);
        basicOperationLayout->addWidget(subtractButton);
        basicOperationLayout->addWidget(multiplyButton);
        basicOperationLayout->addWidget(insertToMatrix1Button);
        basicOperationLayout->addWidget(insertToMatrix2Button);

        // Advanced Operations Tab
        QWidget *advancedOperationsWidget = new QWidget();
        QVBoxLayout *advancedOperationLayout = new QVBoxLayout(advancedOperationsWidget);
        QPushButton *solveSystemButton = new QPushButton("Solve System of Equations");
        QPushButton *matrixArithmeticButton = new QPushButton("Matrix Arithmetic");

        advancedOperationLayout->setSpacing(10);
        advancedOperationLayout->setContentsMargins(10, 10, 10, 10);


        advancedOperationLayout->addWidget(solveSystemButton);
        advancedOperationLayout->addWidget(matrixArithmeticButton);

        // Add tabs to tab widget
        operationTabs->addTab(basicOperationsWidget, basicIcon, "Basic Operations");
        operationTabs->addTab(advancedOperationsWidget, advancedIcon, "Advanced Operations");
        operationTabs->addTab(singleMatrixOpsWidget, singleMatrixIcon, "Single Matrix Operations");

        // operationTabs->addTab(basicOperationsWidget, "Basic Operations");
        // operationTabs->addTab(advancedOperationsWidget, "Advanced Operations");
        // operationTabs->addTab(basicOperationsWidget, "Binary Operations");
        // operationTabs->addTab(singleMatrixOpsWidget, "Single Matrix Operations");

        operationTabs->setMaximumHeight(200);

        // Replace previous operation layout with tab widget
        mainLayout->addWidget(operationTabs);

        // Connect buttons to slots
        connect(resize1Button, &QPushButton::clicked, this, &MatrixCalculatorGUI::resizeMatrix1);
        connect(resize2Button, &QPushButton::clicked, this, &MatrixCalculatorGUI::resizeMatrix2);
        connect(addButton, &QPushButton::clicked, this, &MatrixCalculatorGUI::addMatrices);
        connect(subtractButton, &QPushButton::clicked, this, &MatrixCalculatorGUI::subtractMatrices);
        connect(multiplyButton, &QPushButton::clicked, this, &MatrixCalculatorGUI::multiplyMatrices);
        connect(transposeButton, &QPushButton::clicked, this, &MatrixCalculatorGUI::transposeMatrix);
        connect(determinantButton, &QPushButton::clicked, this, &MatrixCalculatorGUI::calculateDeterminant);
        connect(insertToMatrix1Button, &QPushButton::clicked, this, &MatrixCalculatorGUI::insertResultToMatrix1);
        connect(insertToMatrix2Button, &QPushButton::clicked, this, &MatrixCalculatorGUI::insertResultToMatrix2);

        // Matrix 1 Single Operations
        connect(matrix1TransposeButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::Transpose);
        });
        connect(matrix1DeterminantButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::Determinant);
        });
        connect(matrix1InverseButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::Inverse);
        });
        connect(matrix1LUDecompButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::LUDecomposition);
        });
        connect(matrix1EigenvalueButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::Eigenvalue);
        });
        connect(matrix1TraceButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::Trace);
        });
        connect(matrix1PowerButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::MatrixPower);
        });
        connect(matrix1ScalarMultiplyButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::ScalarMultiply);
        });
        connect(matrix1RankButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::Rank);
        });
        connect(matrix1CholeskyButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::CholeskyDecomposition);
        });
        connect(matrix1NormButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::Norm);
        });
        connect(matrix1GramSchmidtButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::GramSchmidt);
        });
        connect(matrix1ConjugateButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::Conjugate);
        });
        connect(matrix1AdjointButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix1Table, SingleMatrixOperation::Adjoint);
        });


        // Matrix 2 Single Operations
        connect(matrix2TransposeButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::Transpose);
        });
        connect(matrix2DeterminantButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::Determinant);
        });
        connect(matrix2InverseButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::Inverse);
        });
        connect(matrix2LUDecompButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::LUDecomposition);
        });
        connect(matrix2EigenvalueButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::Eigenvalue);
        });
        connect(matrix2TraceButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::Trace);
        });
        connect(matrix2PowerButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::MatrixPower);
        });
        connect(matrix2ScalarMultiplyButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::ScalarMultiply);
        });
        connect(matrix2RankButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::Rank);
        });
        connect(matrix2CholeskyButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::CholeskyDecomposition);
        });
        connect(matrix2NormButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::Norm);
        });
        connect(matrix2GramSchmidtButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::GramSchmidt);
        });
        connect(matrix2ConjugateButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::Conjugate);
        });
        connect(matrix2AdjointButton, &QPushButton::clicked, [this]() {
            performSingleMatrixOperation(matrix2Table, SingleMatrixOperation::Adjoint);
        });

        connect(solveSystemButton, &QPushButton::clicked, this, &MatrixCalculatorGUI::solveSystemOfEquations);
        connect(matrixArithmeticButton, &QPushButton::clicked, this, &MatrixCalculatorGUI::showMatrixArithmeticDialog);
        setupInfoButton();
        connect(infoButton, &QPushButton::clicked, this, &MatrixCalculatorGUI::showAboutDialog);
        setCentralWidget(centralWidget);
        // Create and display the pop-up dialog
        QMessageBox *popup = new QMessageBox(this);
        popup->setWindowTitle("Welcome to Matrix Calculator");

        QString welcomeText = "<div style='text-align: center;'>"
                              "<h2>Matrix Calculator</h2>"
                              "<p><b>Developed by:</b><br>"
                              "Loay Tarek Mostafa 20230298<br>"
                              "Abdelrahman Nabil Hassan 20230219<br>"
                              "Ahmed Ehab Sayed 20230010</p>"
                              "<p><b>Tool made for Math-3 course MA214</b></p>"
                              "<p><b>Supervised by:</b><br>"
                              "Dr. Eng. Moustafa Reda A. Eltantawi</p>"
                              "</div>";

        popup->setText(welcomeText);
        popup->setStyleSheet(
            "QMessageBox {"
            "    background-color: #252526;"
            // "    font-size: 20px;"
            "}"
            "QMessageBox QLabel {"
            "    color: #d4d4d4;"
            "    min-width: 300px;"
             "    font-size: 25px;"
            "}"
            "QMessageBox QPushButton {"
            "    background-color: #0e639c;"
            "    color: white;"
            "    padding: 6px 20px;"
            "    border-radius: 3px;"
            "    min-width: 80px;"
            "}"
            "QMessageBox QPushButton:hover {"
            "    background-color: #1177bb;"
            "}"
            );

        popup->setAttribute(Qt::WA_DeleteOnClose);
        popup->setWindowState(Qt::WindowMaximized);
        popup->show();


    }

private slots:

    void resizeMatrix1() {
        int rows = matrix1RowsSpin->value();
        int cols = matrix1ColsSpin->value();

        // Get existing data to preserve
        vector<vector<double>> existingData = getMatrixFromTable(matrix1Table);

        // Resize matrix table
        matrix1Table->clear();
        matrix1Table->setRowCount(rows);
        matrix1Table->setColumnCount(cols);

        // Restore or initialize data
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = 0.0;
                // Try to preserve existing data if available
                if (i < existingData.size() && j < existingData[i].size()) {
                    value = existingData[i][j];
                }
                QTableWidgetItem* item = new QTableWidgetItem(QString::number(value, 'f', 2));
                matrix1Table->setItem(i, j, item);
            }
        }
    }

    void resizeMatrix2() {
        int rows = matrix2RowsSpin->value();
        int cols = matrix2ColsSpin->value();

        // Get existing data to preserve
        vector<vector<double>> existingData = getMatrixFromTable(matrix2Table);

        // Resize matrix table
        matrix2Table->clear();
        matrix2Table->setRowCount(rows);
        matrix2Table->setColumnCount(cols);

        // Restore or initialize data
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = 0.0;
                // Try to preserve existing data if available
                if (i < existingData.size() && j < existingData[i].size()) {
                    value = existingData[i][j];
                }
                QTableWidgetItem* item = new QTableWidgetItem(QString::number(value, 'f', 2));
                matrix2Table->setItem(i, j, item);
            }
        }
    }

    vector<vector<double>> getMatrixFromTable(QTableWidget* table) {
        vector<vector<double>> matrix;
        vector<double> values;

        for (int i = 0; i < table->rowCount(); i++) {
            vector<double> row;
            for (int j = 0; j < table->columnCount(); j++) {
                QTableWidgetItem* item = table->item(i, j);
                double value = (item && !item->text().isEmpty()) ? item->text().toDouble() : 0.0;
                row.push_back(value);
                values.push_back(value);
            }
            matrix.push_back(row);
        }

        return matrix;
    }

    void setResultTable(const vector<vector<double>>& result) {
        resultTable->clear();
        resultTable->setRowCount(result.size());
        resultTable->setColumnCount(result[0].size());

        for (size_t i = 0; i < result.size(); i++) {
            for (size_t j = 0; j < result[i].size(); j++) {
                QTableWidgetItem* item = new QTableWidgetItem(QString::number(result[i][j], 'f', 2));
                resultTable->setItem(i, j, item);
            }
        }
    }

    void addMatrices() {
        auto matrix1 = getMatrixFromTable(matrix1Table);
        auto matrix2 = getMatrixFromTable(matrix2Table);

        try {
            auto result = MatrixOperations::addMatrix(matrix1, matrix2);
            if (result.empty()) {
                QMessageBox::warning(this, "Error", "Matrices must have the same dimensions for addition.");
                return;
            }
            setResultTable(result);
        } catch (const exception& e) {
            QMessageBox::critical(this, "Error", e.what());
        }
    }

    void subtractMatrices() {
        auto matrix1 = getMatrixFromTable(matrix1Table);
        auto matrix2 = getMatrixFromTable(matrix2Table);

        try {
            auto result = MatrixOperations::subtractMatrix(matrix1, matrix2);
            if (result.empty()) {
                QMessageBox::warning(this, "Error", "Matrices must have the same dimensions for subtraction.");
                return;
            }
            setResultTable(result);
        } catch (const exception& e) {
            QMessageBox::critical(this, "Error", e.what());
        }
    }

    void multiplyMatrices() {
        auto matrix1 = getMatrixFromTable(matrix1Table);
        auto matrix2 = getMatrixFromTable(matrix2Table);

        try {
            auto result = MatrixOperations::multiplyMatrix(matrix1, matrix2);
            if (result.empty()) {
                QMessageBox::warning(this, "Error", "Matrices cannot be multiplied. Check dimensions.");
                return;
            }
            setResultTable(result);
        } catch (const exception& e) {
            QMessageBox::critical(this, "Error", e.what());
        }
    }

    void transposeMatrix() {
        auto matrix1 = getMatrixFromTable(matrix1Table);

        try {
            auto result = MatrixOperations::matrixTranspose(matrix1);
            setResultTable(result);
        } catch (const exception& e) {
            QMessageBox::critical(this, "Error", e.what());
        }
    }

    void calculateDeterminant() {
        auto matrix1 = getMatrixFromTable(matrix1Table);

        try {
            if (matrix1.size() != matrix1[0].size()) {
                QMessageBox::warning(this, "Error", "Determinant can only be calculated for square matrices.");
                return;
            }
            double det = MatrixOperations::matrixDeterminant(matrix1);

            // Clear result table and show determinant
            resultTable->clear();
            resultTable->setRowCount(1);
            resultTable->setColumnCount(1);
            QTableWidgetItem* item = new QTableWidgetItem(QString::number(det, 'f', 4));
            resultTable->setItem(0, 0, item);
        } catch (const exception& e) {
            QMessageBox::critical(this, "Error", e.what());
        }
    }

    void calculateLUDecomposition() {
        auto matrix1 = getMatrixFromTable(matrix1Table);

        try {
            if (matrix1.size() != matrix1[0].size()) {
                QMessageBox::warning(this, "Error", "LU Decomposition requires a square matrix.");
                return;
            }

            auto [L, U] = AdvancedMatrixOperations::doolittleLUDecomposition(matrix1);

            // Create a dialog to show L and U matrices
            QDialog *decompositionDialog = new QDialog(this);
            decompositionDialog->setWindowTitle("LU Decomposition");
            QVBoxLayout *dialogLayout = new QVBoxLayout(decompositionDialog);

            // L Matrix Display
            QLabel *lLabel = new QLabel("Lower Triangular Matrix (L):");
            QTableWidget *lTable = new QTableWidget(L.size(), L[0].size());
            for (size_t i = 0; i < L.size(); i++) {
                for (size_t j = 0; j < L[i].size(); j++) {
                    QTableWidgetItem* item = new QTableWidgetItem(QString::number(L[i][j], 'f', 4));
                    lTable->setItem(i, j, item);
                }
            }
            dialogLayout->addWidget(lLabel);
            dialogLayout->addWidget(lTable);

            // U Matrix Display
            QLabel *uLabel = new QLabel("Upper Triangular Matrix (U):");
            QTableWidget *uTable = new QTableWidget(U.size(), U[0].size());
            for (size_t i = 0; i < U.size(); i++) {
                for (size_t j = 0; j < U[i].size(); j++) {
                    QTableWidgetItem* item = new QTableWidgetItem(QString::number(U[i][j], 'f', 4));
                    uTable->setItem(i, j, item);
                }
            }
            dialogLayout->addWidget(uLabel);
            dialogLayout->addWidget(uTable);

            decompositionDialog->resize(400, 600);
            decompositionDialog->show();
        } catch (const exception& e) {
            QMessageBox::critical(this, "Error", e.what());
        }
    }

    void calculateMatrixInverse() {
        auto matrix1 = getMatrixFromTable(matrix1Table);

        try {
            auto result = AdvancedMatrixOperations::matrixInverse(matrix1);
            setResultTable(result);
        } catch (const exception& e) {
            QMessageBox::warning(this, "Matrix Inverse", e.what());
        }
    }

    void calculateEigenvaluesAndVectors() {
        auto matrix1 = getMatrixFromTable(matrix1Table);

        try {
            // Create a dynamic Eigen matrix based on input matrix size
            int n = matrix1.size();
            Eigen::MatrixXd eigenMatrix(n, n);

            // Transfer data from input matrix to Eigen matrix
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    eigenMatrix(i, j) = matrix1[i][j];
                }
            }

            // Create a dialog to show detailed eigenvalue information
            QDialog *eigenDialog = new QDialog(this);
            eigenDialog->setWindowTitle("Eigenvalues and Eigenvectors");
            QVBoxLayout *dialogLayout = new QVBoxLayout(eigenDialog);

            // Compute the eigenvalues and eigenvectors
            Eigen::EigenSolver<Eigen::MatrixXd> solver(eigenMatrix);

            // Extract real parts of eigenvalues and eigenvectors
            Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
            Eigen::MatrixXd eigenvectors = solver.eigenvectors().real();

            // Eigenvalues Table
            QLabel *eigenValuesLabel = new QLabel("Eigenvalues:");
            dialogLayout->addWidget(eigenValuesLabel);

            QTableWidget *eigenValuesTable = new QTableWidget(n, 1);
            eigenValuesTable->setHorizontalHeaderLabels({"Value"});

            for (int i = 0; i < n; ++i) {
                QTableWidgetItem* item = new QTableWidgetItem(QString::number(eigenvalues(i), 'f', 6));
                eigenValuesTable->setItem(i, 0, item);
            }
            dialogLayout->addWidget(eigenValuesTable);

            // Eigenvectors Table
            QLabel *eigenVectorsLabel = new QLabel("Eigenvectors (columns):");
            dialogLayout->addWidget(eigenVectorsLabel);

            QTableWidget *eigenVectorsTable = new QTableWidget(n, n);
            for (int col = 0; col < n; ++col) {
                eigenVectorsTable->setHorizontalHeaderItem(col, new QTableWidgetItem(QString(" %1").arg(col + 1)));
                for (int row = 0; row < n; ++row) {
                    QTableWidgetItem* item = new QTableWidgetItem(QString::number(eigenvectors(row, col), 'f', 6));
                    eigenVectorsTable->setItem(row, col, item);
                }
            }
            dialogLayout->addWidget(eigenVectorsTable);

            // Additional Mathematical Details
            QTextEdit *detailsText = new QTextEdit();
            detailsText->setReadOnly(true);

            QString details = "<h3>Eigenvalue and Eigenvector Details</h3>";
            details += "<p>For an eigenvalue λ and corresponding eigenvector v, the equation Av = λv holds.</p>";
            details += "<ul>";
            details += "<li>Total Eigenvalues: " + QString::number(n) + "</li>";

            // Calculate some additional properties
            double trace = eigenMatrix.trace();
            double determinant = eigenMatrix.determinant();

            details += "<li>Matrix Trace: " + QString::number(trace, 'f', 6) + "</li>";
            details += "<li>Matrix Determinant: " + QString::number(determinant, 'f', 6) + "</li>";
            details += "</ul>";

            detailsText->setHtml(details);
            dialogLayout->addWidget(detailsText);

            // Close button
            QPushButton *closeButton = new QPushButton("Close");
            connect(closeButton, &QPushButton::clicked, eigenDialog, &QDialog::close);
            dialogLayout->addWidget(closeButton);

            eigenDialog->setLayout(dialogLayout);
            eigenDialog->resize(600, 800);  // Adjust size as needed
            eigenDialog->exec();

        } catch (const exception& e) {
            QMessageBox::warning(this, "Eigenvalue Calculation",
                                 QString("Error calculating eigenvalues: %1").arg(e.what()));
        }
    }

    void insertResultToMatrix1() {
        vector<vector<double>> resultMatrix = getMatrixFromTable(resultTable);

        // Resize Matrix 1 to match result matrix dimensions
        matrix1Table->setRowCount(resultMatrix.size());
        matrix1Table->setColumnCount(resultMatrix[0].size());

        // Populate Matrix 1 with result matrix values
        for (size_t i = 0; i < resultMatrix.size(); i++) {
            for (size_t j = 0; j < resultMatrix[i].size(); j++) {
                QTableWidgetItem* item = new QTableWidgetItem(QString::number(resultMatrix[i][j], 'f', 2));
                matrix1Table->setItem(i, j, item);
            }
        }
    }

    void insertResultToMatrix2() {
        vector<vector<double>> resultMatrix = getMatrixFromTable(resultTable);

        // Resize Matrix 2 to match result matrix dimensions
        matrix2Table->setRowCount(resultMatrix.size());
        matrix2Table->setColumnCount(resultMatrix[0].size());

        // Populate Matrix 2 with result matrix values
        for (size_t i = 0; i < resultMatrix.size(); i++) {
            for (size_t j = 0; j < resultMatrix[i].size(); j++) {
                QTableWidgetItem* item = new QTableWidgetItem(QString::number(resultMatrix[i][j], 'f', 2));
                matrix2Table->setItem(i, j, item);
            }
        }
    }

    // Helper method to convert QTableWidget to vector<vector<double>>
    vector<vector<double>> convertTableToMatrix(QTableWidget* table) {
        vector<vector<double>> matrix;
        int rows = table->rowCount();
        int cols = table->columnCount();

        for (int i = 0; i < rows; ++i) {
            vector<double> row;
            for (int j = 0; j < cols; ++j) {
                QTableWidgetItem* item = table->item(i, j);
                row.push_back(item ? item->text().toDouble() : 0.0);
            }
            matrix.push_back(row);
        }
        return matrix;
    }

    // Helper method to convert a column from QTableWidget to vector<double>
    vector<double> convertTableColumnToVector(QTableWidget* table, int column) {
        vector<double> vec;
        int rows = table->rowCount();

        for (int i = 0; i < rows; ++i) {
            QTableWidgetItem* item = table->item(i, column);
            vec.push_back(item ? item->text().toDouble() : 0.0);
        }
        return vec;
    }

    // Utility function to parse complex number from string
    complex<double> parseComplexNumber(const QString& str) {
        if (str.isEmpty()) return {0, 0};

        QString cleanStr = str.simplified().replace(" ", "");

        // Pure real number
        bool ok;
        double realVal = cleanStr.toDouble(&ok);
        if (ok) return {realVal, 0};

        // Pure imaginary number
        if (cleanStr.endsWith('i')) {
            QString imagStr = cleanStr.chopped(1);
            if (imagStr.isEmpty() || imagStr == "+") return {0, 1};
            if (imagStr == "-") return {0, -1};

            double imagVal = imagStr.toDouble(&ok);
            if (ok) return {0, imagVal};
        }

        // Mixed number (e.g., a+bi)
        QString realPart = cleanStr.section(QRegularExpression("[+-](?=.*i)"), 0, 0);
        QString imagPart = cleanStr.section(QRegularExpression("[+-](?=.*i)"), 1);

        double real = realPart.toDouble(&ok);
        double imag = 0;

        if (!imagPart.isEmpty()) {
            imagPart.remove('i');
            imag = imagPart.toDouble(&ok);
        }

        return {real, imag};
    }

    // Updated methods in MatrixCalculatorGUI to support complex matrices
    vector<vector<complex<double>>> getComplexMatrixFromTable(QTableWidget* table) {
        vector<vector<complex<double>>> matrix;

        for (int i = 0; i < table->rowCount(); i++) {
            vector<complex<double>> row;
            for (int j = 0; j < table->columnCount(); j++) {
                QTableWidgetItem* item = table->item(i, j);
                complex<double> value = item ?
                                            parseComplexNumber(item->text()) :
                                            complex<double>(0, 0);
                row.push_back(value);
            }
            matrix.push_back(row);
        }

        return matrix;
    }

    QString formatComplexNumber(const complex<double>& val) {
        // Handle zero cases
        if (val == complex<double>(0, 0)) return "0";

        // Determine real and imaginary parts
        double real = val.real();
        double imag = val.imag();

        // Case handling for different complex number representations
        // Pure real number
        if (abs(imag) < 1e-10) return QString::number(real);

        // Pure imaginary number
        if (abs(real) < 1e-10) {
            if (imag == 1) return "i";
            if (imag == -1) return "-i";
            return QString::number(imag) + "i";
        }

        // Mixed complex number
        QString result;

        // Add real part
        result += QString::number(real);

        // Add imaginary part with appropriate sign
        if (imag >= 0) {
            result += "+";

            // Special cases for 1 and -1 imaginary coefficients
            if (imag == 1) result += "i";
            else if (imag == -1) result += "-i";
            else result += QString::number(imag) + "i";
        } else {
            // Negative imaginary part
            if (imag == -1) result += "-i";
            else result += QString::number(imag) + "i";
        }

        return result;
    }

    void solveSystemOfEquations() {
        // Convert matrix 1 to LHS (left-hand side) of the equation
        // Convert matrix 2 as RHS (right-hand side)
        vector<vector<double>> LHS = convertTableToMatrix(matrix1Table);
        vector<double> RHS = convertTableColumnToVector(matrix2Table, 0);

        // Validate input matrices
        if (LHS.empty() || RHS.empty() || LHS.size() != RHS.size()) {
            QMessageBox::warning(this, "Invalid Input",
                                 "Ensure Matrix 1 is square and Matrix 2 has a single column matching Matrix 1's dimensions.");
            return;
        }

        // Solve the system
        vector<double> solution = MatrixOperations::solveSystem(LHS, RHS);

        if (solution.empty()) {
            QMessageBox::warning(this, "No Solution",
                                 "The system of equations has no unique solution (singular matrix).");
            return;
        }

        // Create detailed popup
        QDialog *detailDialog = new QDialog(this);
        detailDialog->setWindowTitle("System of Equations Solution");
        detailDialog->setMinimumSize(500, 400);

        QVBoxLayout *dialogLayout = new QVBoxLayout(detailDialog);

        // Rich text widget to support mathematical symbols
        QTextEdit *detailsText = new QTextEdit();
        detailsText->setReadOnly(true);

        // Construct detailed explanation with mathematical notation
        QString explanation = "<html><body>";
        explanation += "<h2>System of Equations Solution</h2>";

        // Display the system of equations in standard mathematical notation
        explanation += "<h3>Original System:</h3>";
        explanation += "<p><b>Coefficient Matrix (A):</b><br>";
        explanation += "A = ⎡";
        for (size_t i = 0; i < LHS.size(); ++i) {
            explanation += "⎢";
            for (size_t j = 0; j < LHS[i].size(); ++j) {
                explanation += QString::number(LHS[i][j], 'f', 4) + " ";
            }
            explanation += "⎥<br>";
        }
        explanation += "⎣</p>";

        // Right-hand side vector
        explanation += "<p><b>Constants Vector (b):</b><br>";
        explanation += "b = ⎡";
        for (double val : RHS) {
            explanation += QString::number(val, 'f', 4) + " ";
        }
        explanation += "⎤</p>";

        // System representation
        explanation += "<h3>Equation Representation:</h3>";
        explanation += "<p>";
        for (size_t i = 0; i < LHS.size(); ++i) {
            for (size_t j = 0; j < LHS[i].size(); ++j) {
                explanation += QString("%1x<sub>%2</sub>").arg(LHS[i][j], 0, 'f', 4).arg(j+1);
                if (j < LHS[i].size() - 1) explanation += " + ";
            }
            explanation += " = " + QString::number(RHS[i], 'f', 4) + "<br>";
        }
        explanation += "</p>";

        // Solution vector
        explanation += "<h3>Solution Vector (x):</h3>";
        explanation += "<p>";
        for (size_t i = 0; i < solution.size(); ++i) {
            explanation += QString("x<sub>%1</sub> = %2<br>").arg(i+1).arg(solution[i], 0, 'f', 4);
        }
        explanation += "</p>";

        // Mathematical symbols and details
        explanation += "<h3>Mathematical Details:</h3>";
        explanation += "<ul>";
        explanation += "<li>∥A∥ (Matrix Norm): Represents the magnitude of the coefficient matrix</li>";
        explanation += "<li>det(A): Determinant of the coefficient matrix</li>";
        explanation += "<li>Ax = b: Linear system representation</li>";
        explanation += "<li>x = A<sup>-1</sup>b: Solution method</li>";
        explanation += "</ul>";

        explanation += "</body></html>";

        detailsText->setHtml(explanation);
        dialogLayout->addWidget(detailsText);

        // Solution table with symbols
        QTableWidget *solutionTable = new QTableWidget(solution.size(), 2);
        solutionTable->setHorizontalHeaderLabels({"Variable (x)", "Value"});
        for (size_t i = 0; i < solution.size(); ++i) {
            // Variable symbol
            QTableWidgetItem *varItem = new QTableWidgetItem(QString("x%1").arg(i+1));
            varItem->setFlags(varItem->flags() & ~Qt::ItemIsEditable);

            // Value
            QTableWidgetItem *valueItem = new QTableWidgetItem(QString::number(solution[i], 'f', 4));
            valueItem->setFlags(valueItem->flags() & ~Qt::ItemIsEditable);

            solutionTable->setItem(i, 0, varItem);
            solutionTable->setItem(i, 1, valueItem);
        }
        solutionTable->resizeColumnsToContents();
        dialogLayout->addWidget(solutionTable);

        // Symbolic representation of solution method
        QLabel *methodLabel = new QLabel("Solution Method: x = A⁻¹b");
        methodLabel->setStyleSheet("font-weight: bold; font-size: 12pt;");
        dialogLayout->addWidget(methodLabel);

        // Close button
        QPushButton *closeButton = new QPushButton("Close");
        connect(closeButton, &QPushButton::clicked, detailDialog, &QDialog::close);
        dialogLayout->addWidget(closeButton);

        detailDialog->setLayout(dialogLayout);
        detailDialog->exec();
    }

    void showMatrixArithmeticDialog() {
        // Create dialog
        QDialog *arithmeticDialog = new QDialog(this);
        arithmeticDialog->setWindowTitle("Matrix Arithmetic Operation");
        QVBoxLayout *dialogLayout = new QVBoxLayout(arithmeticDialog);

        // Operation input
        QLabel *operationLabel = new QLabel("Enter Operation (e.g., 2*A + 3*B):");
        QLineEdit *operationLineEdit = new QLineEdit();
        operationLineEdit->setPlaceholderText("2*A + 3*B");

        // Result display
        QTableWidget *resultTable = new QTableWidget();

        // Calculate button
        QPushButton *calculateButton = new QPushButton("Calculate");

        // Add widgets to layout
        dialogLayout->addWidget(operationLabel);
        dialogLayout->addWidget(operationLineEdit);
        dialogLayout->addWidget(calculateButton);

        // Create a widget to hold the result table
        QWidget *resultWidget = new QWidget();
        QVBoxLayout *resultLayout = new QVBoxLayout(resultWidget);
        resultLayout->addWidget(resultTable);
        dialogLayout->addWidget(resultWidget);

        // Connect calculate button
        connect(calculateButton, &QPushButton::clicked, [this, operationLineEdit, resultTable]() {
            try {
                string operation = operationLineEdit->text().toStdString();

                // Get matrices from tables
                auto matrix1 = getMatrixFromTable(matrix1Table);
                auto matrix2 = getMatrixFromTable(matrix2Table);

                // Perform arithmetic operation
                auto resultMatrix = AdvancedMatrixOperations::performMatrixArithmeticOperation(
                    matrix1, matrix2, operation
                    );

                // Set up result table
                resultTable->clear();
                resultTable->setRowCount(resultMatrix.size());
                resultTable->setColumnCount(resultMatrix[0].size());

                // Populate result table
                for (size_t i = 0; i < resultMatrix.size(); ++i) {
                    for (size_t j = 0; j < resultMatrix[i].size(); ++j) {
                        QTableWidgetItem* item = new QTableWidgetItem(
                            QString::number(resultMatrix[i][j], 'f', 4)
                            );
                        resultTable->setItem(i, j, item);
                    }
                }
            } catch (const exception& e) {
                QMessageBox::critical(this, "Error", e.what());
            }
        });

        // Resize and show dialog
        arithmeticDialog->resize(400, 500);
        arithmeticDialog->show();
    }

    void setupInfoButton() {
        infoButton = new QPushButton("", this);
        infoButton->setFixedSize(20, 20);
        infoButton->setCursor(Qt::PointingHandCursor);  // Changes cursor to hand when hovering

        // Create a more sophisticated style for the button
        infoButton->setStyleSheet(
            "QPushButton {"
            "    background-color: #0e639c;"
            "    color: white;"
            "    border-radius: 10px;"  // Make it perfectly circular
            "    font-family: 'Segoe UI', Arial;"
            "    font-size: 15px;"
            "    font-weight: bold;"
            "    text-align: center;"
            "    border: 2px solid #0e639c;"  // Initial border color
            "    margin: 0;"
            "    padding: 0;"
            "}"
            "QPushButton:hover {"
            "    background-color: #1177bb;"
            "    border: 2px solid #1177bb;"
            "    color: white;"
            "}"
            "QPushButton:pressed {"
            "    background-color: #094771;"
            "    border: 2px solid #094771;"
            "}"
            );

        // Create a custom icon using Unicode character
        QFont iconFont = infoButton->font();
        iconFont.setFamily("Segoe UI");  // Use Segoe UI for better icon rendering
        iconFont.setPixelSize(15);
        iconFont.setBold(true);
        infoButton->setFont(iconFont);
        infoButton->setText("i");

        // Add a tool tip
        infoButton->setToolTip("About Matrix Calculator");

        // Position the button in the top right corner with some padding
        infoButton->move(width() - 45, 2);

        // Optional: Add a subtle shadow effect
        QGraphicsDropShadowEffect* shadow = new QGraphicsDropShadowEffect;
        shadow->setBlurRadius(8);
        shadow->setColor(QColor(0, 0, 0, 80));
        shadow->setOffset(0, 2);
        infoButton->setGraphicsEffect(shadow);

        // Make sure the button stays in the right position when window is resized
        connect(this, &QMainWindow::windowTitleChanged, [this]() {
            infoButton->move(width() - 45, 2);
        });


    }

    void resizeEvent(QResizeEvent* event) {
        QMainWindow::resizeEvent(event);
        if (infoButton) {
            infoButton->move(width() - 45, 2);
        }
    }

    void showAboutDialog() {
        QMessageBox* popup = new QMessageBox(this);
        popup->setWindowTitle("Welcome to Matrix Calculator");
        QString welcomeText = "<div style='text-align: center;'>"
                              "<h2>Matrix Calculator</h2>"
                              "<p><b>Developed by:</b><br>"
                              "Loay Tarek Mostafa 20230298<br>"
                              "Abdelrahman Nabil Hassan 20230219<br>"
                              "Ahmed Ehab Sayed 20230010</p>"
                              "<p><b>Tool made for Math-3 course MA214</b></p>"
                              "<p><b>Supervised by:</b><br>"
                              "Dr. Eng. Moustafa Reda El-Tantawi</p>"
                              "</div>";
        popup->setText(welcomeText);
        popup->setStyleSheet(
            "QMessageBox {"
            "    background-color: #252526;"
            "}"
            "QMessageBox QLabel {"
            "    color: #d4d4d4;"
            "    min-width: 300px;"
             "    font-size: 25px;"
            "}"
            "QMessageBox QPushButton {"
            "    background-color: #0e639c;"
            "    color: white;"
            "    padding: 6px 20px;"
            "    border-radius: 3px;"
            "    min-width: 80px;"
            "}"
            "QMessageBox QPushButton:hover {"
            "    background-color: #1177bb;"
            "}"
            );
        popup->setAttribute(Qt::WA_DeleteOnClose);
        popup->setWindowState(Qt::WindowMaximized);
        popup->show();
    }

private:
    // Enum to specify single matrix operation type
    enum class SingleMatrixOperation {
        Transpose,
        Determinant,
        Inverse,
        LUDecomposition,
        Eigenvalue,
        Trace,
        MatrixPower,
        ScalarMultiply,
        Conjugate,
        Adjoint,
        Rank,
        CholeskyDecomposition,
        Norm,
        GramSchmidt
    };

    // Generic method to perform single matrix operations
    void performSingleMatrixOperation(QTableWidget* sourceTable, SingleMatrixOperation operation) {
        auto matrix = getMatrixFromTable(sourceTable);

        try {
            switch (operation) {
            case SingleMatrixOperation::Transpose: {
                auto result = MatrixOperations::matrixTranspose(matrix);
                setResultTable(result);
                break;
            }
            case SingleMatrixOperation::Determinant: {
                if (matrix.size() != matrix[0].size()) {
                    QMessageBox::warning(this, "Error", "Determinant requires a square matrix.");
                    return;
                }
                double det = MatrixOperations::matrixDeterminant(matrix);

                // Clear result table and show determinant
                resultTable->clear();
                resultTable->setRowCount(1);
                resultTable->setColumnCount(1);
                QTableWidgetItem* item = new QTableWidgetItem(QString::number(det, 'f', 4));
                resultTable->setItem(0, 0, item);
                break;
            }
            case SingleMatrixOperation::Inverse: {
                auto result = AdvancedMatrixOperations::matrixInverse(matrix);
                setResultTable(result);
                break;
            }
            case SingleMatrixOperation::LUDecomposition: {
                if (matrix.size() != matrix[0].size()) {
                    QMessageBox::warning(this, "Error", "LU Decomposition requires a square matrix.");
                    return;
                }

                // Ask user to choose a method (Doolittle or Crout)
                QDialog *methodDialog = new QDialog(this);
                methodDialog->setWindowTitle("Select LU Decomposition Method");
                methodDialog->setMinimumWidth(400);  // Set minimum width
                methodDialog->setMinimumHeight(300); // Set minimum height

                QVBoxLayout *layout = new QVBoxLayout(methodDialog);
                layout->setSpacing(20);              // Increase spacing between widgets
                layout->setContentsMargins(30, 30, 30, 30); // Add margins around the layout

                QLabel *instructionLabel = new QLabel("Please select the LU decomposition method:");
                instructionLabel->setStyleSheet("font-size: 14pt; margin-bottom: 10px;");

                QRadioButton *doolittleButton = new QRadioButton("Doolittle Method");
                doolittleButton->setStyleSheet("font-size: 12pt; padding: 10px;");

                QRadioButton *croutButton = new QRadioButton("Crout Method");
                croutButton->setStyleSheet("font-size: 12pt; padding: 10px;");

                QPushButton *okButton = new QPushButton("OK");
                okButton->setMinimumHeight(40);      // Make button taller
                okButton->setStyleSheet("font-size: 12pt; padding: 5px 20px;");

                layout->addWidget(instructionLabel);
                layout->addWidget(doolittleButton);
                layout->addWidget(croutButton);
                layout->addSpacing(20);              // Add extra space before the button
                layout->addWidget(okButton);

                connect(okButton, &QPushButton::clicked, methodDialog, &QDialog::accept);
                methodDialog->exec();

                if (!doolittleButton->isChecked() && !croutButton->isChecked()) {
                    QMessageBox::warning(this, "Error", "You must select a decomposition method.");
                    delete methodDialog;
                    return;
                }

                // Perform selected decomposition
                std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> decompositionResult;

                if (doolittleButton->isChecked()) {
                    decompositionResult = AdvancedMatrixOperations::doolittleLUDecomposition(matrix);
                } else if (croutButton->isChecked()) {
                    decompositionResult = AdvancedMatrixOperations::crout_decomposition(matrix);
                }

                delete methodDialog;

                auto [L, U] = decompositionResult;

                // Create a dialog to display L and U matrices
                QDialog *decompositionDialog = new QDialog(this);
                decompositionDialog->setWindowTitle("LU Decomposition Result");
                QVBoxLayout *dialogLayout = new QVBoxLayout(decompositionDialog);

                // L Matrix Display
                QLabel *lLabel = new QLabel("Lower Triangular Matrix (L):");
                QTableWidget *lTable = new QTableWidget(L.size(), L[0].size());
                for (size_t i = 0; i < L.size(); i++) {
                    for (size_t j = 0; j < L[i].size(); j++) {
                        QTableWidgetItem* item = new QTableWidgetItem(QString::number(L[i][j], 'f', 4));
                        lTable->setItem(i, j, item);
                    }
                }
                dialogLayout->addWidget(lLabel);
                dialogLayout->addWidget(lTable);

                // U Matrix Display
                QLabel *uLabel = new QLabel("Upper Triangular Matrix (U):");
                QTableWidget *uTable = new QTableWidget(U.size(), U[0].size());
                for (size_t i = 0; i < U.size(); i++) {
                    for (size_t j = 0; j < U[i].size(); j++) {
                        QTableWidgetItem* item = new QTableWidgetItem(QString::number(U[i][j], 'f', 4));
                        uTable->setItem(i, j, item);
                    }
                }
                dialogLayout->addWidget(uLabel);
                dialogLayout->addWidget(uTable);

                decompositionDialog->resize(400, 600);
                decompositionDialog->show();
                break;
            }
            case SingleMatrixOperation::Eigenvalue: {
                if (matrix.size() != matrix[0].size()) {
                    QMessageBox::warning(this, "Error", "Eigenvalue requires a square matrix.");
                    return;
                }

                // Create a dialog to show detailed eigenvalue information
                QDialog *eigenDialog = new QDialog(this);
                eigenDialog->setWindowTitle("Eigenvalues and Eigenvectors");
                QVBoxLayout *dialogLayout = new QVBoxLayout(eigenDialog);

                try {
                    // Create a dynamic Eigen matrix based on input matrix size
                    int n = matrix.size();
                    Eigen::MatrixXd eigenMatrix(n, n);

                    // Transfer data from input matrix to Eigen matrix
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < n; ++j) {
                            eigenMatrix(i, j) = matrix[i][j];
                        }
                    }

                    // Compute the eigenvalues and eigenvectors
                    Eigen::EigenSolver<Eigen::MatrixXd> solver(eigenMatrix);

                    // Extract real parts of eigenvalues and eigenvectors
                    Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
                    Eigen::MatrixXd eigenvectors = solver.eigenvectors().real();

                    // Eigenvalues Table
                    QLabel *eigenValuesLabel = new QLabel("<h3>Eigenvalues:</h3>");
                    eigenValuesLabel->setTextFormat(Qt::RichText);
                    dialogLayout->addWidget(eigenValuesLabel);

                    QTableWidget *eigenValuesTable = new QTableWidget(n, 2);
                    eigenValuesTable->setHorizontalHeaderLabels({"Index", "Value"});

                    for (int i = 0; i < n; ++i) {
                        // Index column
                        QTableWidgetItem* indexItem = new QTableWidgetItem(QString::number(i + 1));
                        eigenValuesTable->setItem(i, 0, indexItem);

                        // Value column
                        QTableWidgetItem* valueItem = new QTableWidgetItem(QString::number(eigenvalues(i), 'f', 6));
                        eigenValuesTable->setItem(i, 1, valueItem);
                    }
                    dialogLayout->addWidget(eigenValuesTable);

                    // Eigenvectors Table
                    QLabel *eigenVectorsLabel = new QLabel("<h3>Eigenvectors (columns):</h3>");
                    eigenVectorsLabel->setTextFormat(Qt::RichText);
                    dialogLayout->addWidget(eigenVectorsLabel);

                    QTableWidget *eigenVectorsTable = new QTableWidget(n, n + 1);

                    // Set headers
                    eigenVectorsTable->setHorizontalHeaderItem(0, new QTableWidgetItem("Eigenvalue"));
                    for (int col = 0; col < n; ++col) {
                        eigenVectorsTable->setHorizontalHeaderItem(col + 1, new QTableWidgetItem(QString(" %1").arg(col + 1)));
                    }

                    for (int col = 0; col < n; ++col) {
                        // First column: corresponding eigenvalue
                        QTableWidgetItem* eigenvalueItem = new QTableWidgetItem(QString::number(eigenvalues(col), 'f', 6));
                        eigenVectorsTable->setItem(col, 0, eigenvalueItem);

                        // Subsequent columns: eigenvector components
                        for (int row = 0; row < n; ++row) {
                            QTableWidgetItem* item = new QTableWidgetItem(QString::number(eigenvectors(row, col), 'f', 6));
                            eigenVectorsTable->setItem(col, row + 1, item);
                        }
                    }
                    dialogLayout->addWidget(eigenVectorsTable);

                    // Additional Mathematical Details
                    QTextEdit *detailsText = new QTextEdit();
                    detailsText->setReadOnly(true);

                    QString details = "<h3>Eigenvalue and Eigenvector Details</h3>";
                    details += "<p>For an eigenvalue λ and corresponding eigenvector v, the equation Av = λv holds.</p>";
                    details += "<ul>";
                    details += "<li>Total Eigenvalues: " + QString::number(n) + "</li>";

                    // Calculate some additional properties
                    double trace = eigenMatrix.trace();
                    double determinant = eigenMatrix.determinant();

                    details += "<li>Matrix Trace: " + QString::number(trace, 'f', 6) + "</li>";
                    details += "<li>Matrix Determinant: " + QString::number(determinant, 'f', 6) + "</li>";
                    details += "</ul>";

                    detailsText->setHtml(details);
                    dialogLayout->addWidget(detailsText);

                    // Close button
                    QPushButton *closeButton = new QPushButton("Close");
                    connect(closeButton, &QPushButton::clicked, eigenDialog, &QDialog::close);
                    dialogLayout->addWidget(closeButton);

                    eigenDialog->setLayout(dialogLayout);
                    eigenDialog->resize(700, 900);  // Adjusted size for more comprehensive display
                    eigenDialog->exec();

                } catch (const exception& e) {
                    QMessageBox::warning(this, "Eigenvalue Calculation",
                                         QString("Error calculating eigenvalues: %1").arg(e.what()));
                }
                break;
            }
            case SingleMatrixOperation::Trace: {
                double trace = AdvancedMatrixOperations::calculateTrace(matrix);
                resultTable->clear();
                resultTable->setRowCount(1);
                resultTable->setColumnCount(1);
                QTableWidgetItem* item = new QTableWidgetItem(QString::number(trace, 'f', 4));
                resultTable->setItem(0, 0, item);
                break;
            }
            case SingleMatrixOperation::MatrixPower: {
                bool ok;
                int power = QInputDialog::getInt(this, "Matrix Power",
                                                 "Enter power (integer):", 2, 1, 10, 1, &ok);
                if (ok) {
                    auto result = AdvancedMatrixOperations::calculateMatrixPower(matrix, power);
                    setResultTable(result);
                }
                break;
            }
            case SingleMatrixOperation::ScalarMultiply: {
                bool ok;
                double scalar = QInputDialog::getDouble(this, "Scalar Multiplication",
                                                        "Enter scalar value:", 1.0, -1000, 1000, 4, &ok);
                if (ok) {
                    auto result = AdvancedMatrixOperations::scalarMultiplyMatrix(matrix, scalar);
                    setResultTable(result);
                }
                break;
            }
            case SingleMatrixOperation::Rank: {
                int rank = AdvancedMatrixOperations::calculateMatrixRank(matrix);
                resultTable->clear();
                resultTable->setRowCount(1);
                resultTable->setColumnCount(1);
                QTableWidgetItem* item = new QTableWidgetItem(QString::number(rank));
                resultTable->setItem(0, 0, item);
                break;
            }
            case SingleMatrixOperation::CholeskyDecomposition: {
                // Ensure matrix is square and symmetric
                if (matrix.size() != matrix[0].size()) {
                    QMessageBox::warning(this, "Error", "Cholesky Decomposition requires a square matrix.");
                    return;
                }

                auto [L, U] = AdvancedMatrixOperations::choleskyDecomposition(matrix);

                // Create a dialog to show L and U matrices
                QDialog *decompositionDialog = new QDialog(this);
                decompositionDialog->setWindowTitle("Cholesky Decomposition");
                QVBoxLayout *dialogLayout = new QVBoxLayout(decompositionDialog);

                // L Matrix Display
                QLabel *lLabel = new QLabel("Lower Triangular Matrix (L):");
                QTableWidget *lTable = new QTableWidget(L.size(), L[0].size());
                for (size_t i = 0; i < L.size(); i++) {
                    for (size_t j = 0; j < L[i].size(); j++) {
                        QTableWidgetItem* item = new QTableWidgetItem(QString::number(L[i][j], 'f', 4));
                        lTable->setItem(i, j, item);
                    }
                }
                dialogLayout->addWidget(lLabel);
                dialogLayout->addWidget(lTable);

                // U Matrix Display
                QLabel *uLabel = new QLabel("Upper Triangular Matrix (U):");
                QTableWidget *uTable = new QTableWidget(U.size(), U[0].size());
                for (size_t i = 0; i < U.size(); i++) {
                    for (size_t j = 0; j < U[i].size(); j++) {
                        QTableWidgetItem* item = new QTableWidgetItem(QString::number(U[i][j], 'f', 4));
                        uTable->setItem(i, j, item);
                    }
                }
                dialogLayout->addWidget(uLabel);
                dialogLayout->addWidget(uTable);

                decompositionDialog->resize(400, 600);
                decompositionDialog->show();
                break;
            }
            case SingleMatrixOperation::GramSchmidt: {
                // Create an Eigen matrix from the input matrix
                int rows = matrix.size();
                int cols = matrix[0].size();
                Eigen::MatrixXd eigenMatrix(rows, cols);

                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        eigenMatrix(i, j) = matrix[i][j];
                    }
                }

                // Perform Gram-Schmidt orthonormalization
                Eigen::MatrixXd orthonormalBasis = AdvancedMatrixOperations::gramSchmidt(eigenMatrix);

                // Convert back to vector of vectors for display
                vector<vector<double>> result(rows, vector<double>(cols));
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        result[i][j] = orthonormalBasis(i, j);
                    }
                }

                setResultTable(result);
                break;
            }
            case SingleMatrixOperation::Conjugate: {
                vector<vector<complex<double>>> complexMatrix = getComplexMatrixFromTable(sourceTable);

                vector<vector<complex<double>>> conjugateMatrix;
                for (const auto& row : complexMatrix) {
                    vector<complex<double>> conjugateRow;
                    for (const auto& elem : row) {
                        conjugateRow.push_back(conj(elem));
                    }
                    conjugateMatrix.push_back(conjugateRow);
                }

                // Display conjugate matrix
                resultTable->clear();
                resultTable->setRowCount(conjugateMatrix.size());
                resultTable->setColumnCount(conjugateMatrix[0].size());

                for (size_t i = 0; i < conjugateMatrix.size(); i++) {
                    for (size_t j = 0; j < conjugateMatrix[i].size(); j++) {
                        complex<double> val = conjugateMatrix[i][j];

                        // Use the new formatting function
                        QString displayText = formatComplexNumber(val);

                        QTableWidgetItem* item = new QTableWidgetItem(displayText);
                        resultTable->setItem(i, j, item);
                    }
                }
                break;
            }
            case SingleMatrixOperation::Adjoint: {
                if (matrix.size() != matrix[0].size()) {
                    QMessageBox::warning(this, "Error", "Adjoint requires a square matrix.");
                    return;
                }

                // Create a dynamic Eigen matrix based on input matrix size
                int n = matrix.size();
                Eigen::MatrixXd eigenMatrix(n, n);

                // Transfer data from input matrix to Eigen matrix
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        eigenMatrix(i, j) = matrix[i][j];
                    }
                }

                // Compute the adjoint
                Eigen::MatrixXd adjointMatrix = AdvancedMatrixOperations::computeAdjoint(eigenMatrix);

                // Convert back to vector of vectors for display
                vector<vector<double>> resultMatrix(n, vector<double>(n));
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        resultMatrix[i][j] = adjointMatrix(i, j);
                    }
                }

                setResultTable(resultMatrix);
                break;
            }
            case SingleMatrixOperation::Norm: {
                QDialog *normDialog = new QDialog(this);
                normDialog->setWindowTitle("Choose Norm Type");
                normDialog->setMinimumWidth(450);    // Increased width
                normDialog->setMinimumHeight(300);   // Increased height

                QVBoxLayout *layout = new QVBoxLayout(normDialog);
                layout->setSpacing(20);              // Increase spacing between widgets
                layout->setContentsMargins(30, 30, 30, 30); // Add margins

                QLabel *label = new QLabel("Select Norm Type:");
                label->setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 10px;");

                QComboBox *normTypeCombo = new QComboBox();
                normTypeCombo->setStyleSheet("font-size: 12pt; padding: 8px;");
                normTypeCombo->addItem("Frobenius Norm", 0);
                normTypeCombo->addItem("1-Norm (Max Column Sum)", 1);
                normTypeCombo->addItem("Infinity Norm (Max Row Sum)", 2);
                normTypeCombo->setMinimumHeight(40); // Make combo box taller
                normTypeCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

                QPushButton *calculateButton = new QPushButton("Calculate");
                calculateButton->setMinimumHeight(45);  // Taller button
                calculateButton->setStyleSheet("font-size: 12pt; padding: 8px 20px;");

                layout->addWidget(label);
                layout->addWidget(normTypeCombo);
                layout->addSpacing(20);              // Add extra space before button
                layout->addWidget(calculateButton);
                layout->addStretch();

                connect(calculateButton, &QPushButton::clicked, [this, normDialog, normTypeCombo, matrix]() {
                    int normType = normTypeCombo->currentData().toInt();
                    double norm = AdvancedMatrixOperations::calculateNorm(matrix, normType);

                    resultTable->clear();
                    resultTable->setRowCount(1);
                    resultTable->setColumnCount(1);
                    QTableWidgetItem* item = new QTableWidgetItem(QString::number(norm, 'f', 4));
                    resultTable->setItem(0, 0, item);

                    normDialog->accept();
                });

                normDialog->show();
                break;
            }
            }
        } catch (const exception& e) {
            QMessageBox::critical(this, "Error", e.what());
        }

    }


    QPushButton* infoButton;
    QTableWidget *matrix1Table;
    QTableWidget *matrix2Table;
    QTableWidget *resultTable;
    QSpinBox *matrix1RowsSpin;
    QSpinBox *matrix1ColsSpin;
    QSpinBox *matrix2RowsSpin;
    QSpinBox *matrix2ColsSpin;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    MatrixCalculatorGUI calculator;
    calculator.setWindowState(Qt::WindowMaximized);
    calculator.show();

    return app.exec();
}

#include "main.moc"
