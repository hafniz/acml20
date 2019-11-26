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
        private Table() => throw new InvalidOperationException();
        public Table(List<List<T>> data) => Data = data;

        protected List<List<T>> Data { get; set; }
        public int RowCount => Data.Count;
        public int ColumnCount => Data.FirstOrDefault().Count;

        public List<T> this[int rowIndex] => Data[rowIndex];
        public Table<T> SelectRows(params int[] rowIndexes)
        {
            List<List<T>> data = new List<List<T>>();
            rowIndexes.ToList().ForEach(r => data.Add(Data[r]));
            return new Table<T>(data);
        }
        public static Table<T> JoinRows(params List<T>[] rows) => new Table<T>(rows.ToList());

        public List<T> SelectColumn(int columnIndex)
        {
            List<T> column = new List<T>();
            Data.ForEach(r => column.Add(r[columnIndex]));
            return column;
        }
        public Table<T> SelectColumns(params int[] columnIndexes)
        {
            List<List<T>> data = new List<List<T>>();
            for (int r = 0; r < RowCount; r++)
            {
                List<T> newRow = new List<T>();
                foreach (int c in columnIndexes)
                {
                    newRow.Add(Data[r][c]);
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
        public IEnumerator<List<T>> GetEnumerator() => ((IEnumerable<List<T>>)Data).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable<List<T>>)Data).GetEnumerator();
    }
}
