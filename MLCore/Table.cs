using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MLCore
{
    [DebuggerStepThrough]
    public class Table<T> : IEnumerable<List<T>>
    {
        private readonly List<List<T>> data;
        public int RowCount => data.Count;
        public int ColumnCount => data.FirstOrDefault().Count;

        private Table() => throw new InvalidOperationException();
        public Table(List<List<T>> source) => data = source;

        public List<T> this[Index rowIndex] => data[rowIndex];
        public Table<T> SelectRows(params Index[] rowIndices)
        {
            List<List<T>> data = new List<List<T>>();
            rowIndices.ToList().ForEach(r => data.Add(this.data[r]));
            return new Table<T>(data);
        }
        public static Table<T> JoinRows(params List<T>[] rows) => new Table<T>(rows.ToList());

        public List<T> SelectColumn(Index columnIndex)
        {
            List<T> column = new List<T>();
            data.ForEach(r => column.Add(r[columnIndex]));
            return column;
        }
        public Table<T> SelectColumns(params Index[] columnIndices)
        {
            List<List<T>> data = new List<List<T>>();
            for (int r = 0; r < RowCount; r++)
            {
                List<T> newRow = new List<T>();
                foreach (Index c in columnIndices)
                {
                    newRow.Add(this.data[r][c]);
                }
                data.Add(newRow);
            }
            return new Table<T>(data);
        }
        public static Table<T> JoinColumns(params List<T>[] columns)
        {
            List<List<T>> data = new List<List<T>>();
            for (int r = 0; r < columns.FirstOrDefault().Count; r++)
            {
                List<T> row = new List<T>();
                columns.ToList().ForEach(c => row.Add(c[r]));
                data.Add(row);
            }
            return new Table<T>(data);
        }

        public Table<T> Transpose()
        {
            List<List<T>> data = new List<List<T>>();
            for (int c = 0; c < ColumnCount; c++)
            {
                data.Add(SelectColumn(c));
            }
            return new Table<T>(data);
        }
        public static Table<string> Average(params Table<string>[] tables)
        {
            List<List<string>> data = new List<List<string>>();
            for (int r = 0; r < tables.First().RowCount; r++)
            {
                List<string> avgRow = new List<string>();
                for (int c = 0; c < tables.First().ColumnCount; c++)
                {
                    double sum = tables.Sum(t => double.Parse(t[r][c]));
                    avgRow.Add((sum / tables.Length).ToString());
                }
                data.Add(avgRow);
            }
            return new Table<string>(data);
        }

        public static Table<T> MergeHorizontal(Table<T> left, Table<T> right)
        {
            List<List<T>> mergedData = new List<List<T>>();
            for (int r = 0; r < Math.Min(left.RowCount, right.RowCount); r++)
            {
                mergedData.Add(left[r].Concat(right[r]).ToList());
            }
            if (left.RowCount > right.RowCount)
            {
                for (int r = 0; r < left.RowCount - right.RowCount; r++)
                {
                    mergedData.Add(left[r].Concat(Enumerable.Repeat(default(T), right.ColumnCount)).ToList());
                }
            }
            else if (right.RowCount > left.RowCount)
            {
                for (int r = 0; r < right.RowCount - left.RowCount; r++)
                {
                    mergedData.Add(Enumerable.Repeat(default(T), left.ColumnCount).Concat(right[r]).ToList());
                }
            }

            return new Table<T>(mergedData);
        }

        public IEnumerator<List<T>> GetEnumerator() => ((IEnumerable<List<T>>)data).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable<List<T>>)data).GetEnumerator();
    }
}
