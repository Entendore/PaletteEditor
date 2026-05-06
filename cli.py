#!/usr/bin/env python3
import argparse
import sys
import app

def main():
    parser = argparse.ArgumentParser(description="DuckPalette CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate
    p_gen = subparsers.add_parser("generate", help="Generate random palettes")
    p_gen.add_argument("--count", type=int, default=10, help="Number to generate")

    # Export
    p_exp = subparsers.add_parser("export", help="Export palette to file")
    p_exp.add_argument("id", type=int, help="Palette ID")
    p_exp.add_argument("--out", default="output.map", help="Output file")

    # Import
    p_imp = subparsers.add_parser("import", help="Import file to DB")
    p_imp.add_argument("path", help="File path")

    # Query
    p_query = subparsers.add_parser("query", help="Search DB")
    p_query.add_argument("--min-bright", type=float)
    p_query.add_argument("--dominant", choices=['R', 'G', 'B'])
    
    # Delete
    p_del = subparsers.add_parser("delete", help="Delete palette from DB")
    p_del.add_argument("id", type=int, help="Palette ID to delete")

    args = parser.parse_args()

    if args.command == "generate":
        print(f"Generating {args.count} palettes...")
        app.controller.generate_new_palettes(args.count)
        print("Done.")

    elif args.command == "export":
        success = app.controller.export_db_to_file(args.id, args.out)
        if success:
            print(f"ID {args.id} saved to {args.out}")
        else:
            print("Failed to export (ID not found?).")

    elif args.command == "import":
        pid, _, name = app.controller.import_file_to_db(args.path)
        if pid:
            print(f"Imported '{name}' as ID {pid}")
        else:
            print("Import failed (Empty file?).")

    elif args.command == "query":
        results = app.controller.search_palettes(
            min_bright=args.min_bright, 
            dominant=args.dominant
        )
        print(f"Found {len(results)} palettes:")
        for r in results:
            pid, name, b, c, d, num, _ = r
            print(f"ID:{pid:<5} {name:<20} Bright:{b:.1f} [{d}]")

    elif args.command == "delete":
        app.controller.delete_palette(args.id)
        print(f"Deleted palette ID {args.id}.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()