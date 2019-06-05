using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MLCore
{
    [DebuggerStepThrough]
    public static class TabularOperation
    {
        /// <summary>
        /// Select a single column from a two-dimensional list. 
        /// </summary>
        /// <param name="table">The two-dimensional list from where the data are to be selected. </param>
        /// <param name="columnIndex">The zero-based index of the row to be selected. </param>
        /// <returns>Values in the specified row. </returns>
        public static List<string> SelectColumn(this List<List<string>> table, int columnIndex)
        {
            List<string> column = new List<string>();
            table.ForEach(r => column.Add(r[columnIndex]));
            return column;
        }

        /// <summary>
        /// Join multiple columns into a two-dimensional list. 
        /// </summary>
        /// <param name="columns">List of column values to be joined. </param>
        /// <returns>A two-dimensional list, each element in which represents a ROW of values. </returns>
        public static List<List<string>> JoinColumns(params List<string>[] columns)
        {
            List<List<string>> table = new List<List<string>>();
            for (int r = 0; r < columns[0].Count; r++)
            {
                List<string> row = new List<string>();
                columns.ToList().ForEach(c => row.Add(c[r]));
                table.Add(row);
            }
            return table;
        }

        /// <summary>
        /// Insert a column to an existing two-dimensional list and shift the rest columns rightwards. 
        /// </summary>
        /// <param name="table">The two-dimensional list into which the column is to be added. </param>
        /// <param name="column">The column of values to be added. </param>
        /// <param name="columnIndex">The zero-based index at where the column is to be added; 
        /// i.e., the column will have this index after the operation. </param>
        public static void InsertColumn(this List<List<string>> table, List<string> column, int columnIndex)
        {
            for (int r = 0; r < table.Count; r++)
            {
                table[r].Insert(columnIndex, column[r]);
            }
        }

        /// <summary>
        /// Calculates average values for each corresponding cell for the specified two-dimensional lists. 
        /// </summary>
        /// <param name="tables">The two-dimensional lists that contains values to be calculated for. </param>
        /// <returns>A two-dimensional list of the same size as the source, containing the average values calculated. </returns>
        public static List<List<string>> Average(params List<List<string>>[] tables)
        {
            List<List<string>> averageTable = new List<List<string>>();
            for (int r = 0; r < tables[0].Count; r++)
            {
                List<string> averageRow = new List<string>();
                for (int c = 0; c < tables[0][0].Count; c++)
                {
                    double cellSum = 0;
                    foreach (List<List<string>> table in tables)
                    {
                        cellSum += double.Parse(table[r][c]);
                    }
                    averageRow.Add((cellSum / tables.Length).ToString());
                }
                averageTable.Add(averageRow);
            }
            return averageTable;
        }

        /// <summary>
        /// Transposes a two-dimensional list. 
        /// </summary>
        /// <param name="table">The two-dimensional list to be transposed. </param>
        /// <returns>The transposed two-dimensional list. </returns>
        public static List<List<string>> Transpose(this List<List<string>> table)
        {
            List<List<string>> transposed = new List<List<string>>();
            for (int i = 0; i < table[0].Count; i++)
            {
                transposed.Add(new List<string>());
                for (int j = 0; j < table.Count; j++)
                {
                    transposed[i].Add(table[j][i]);
                }
            }
            return transposed;
        }

        /// <summary>
        /// Calculates the difference between each element in the two lists. 
        /// </summary>
        /// <param name="minuend">A list of numbers that serves as the minuend. </param>
        /// <param name="subtrahend">A list of numbers that serves as the subtrahend. </param>
        /// <returns>A list of numbers whose each element represents the numerical difference between the two corresponding numbers in the arguments. </returns>
        public static List<string> Minus(this List<string> minuend, List<string> subtrahend)
        {
            IEnumerable<string> diff = new List<string>();
            for (int i = 0; i < minuend.Count; i++)
            {
                diff = diff.Append((double.Parse(minuend[i]) - double.Parse(subtrahend[i])).ToString());
            }
            return diff.ToList();
        }
    }
}
